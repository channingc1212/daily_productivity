from typing import Dict, Any, List
from .base import BaseAgent, AgentResponse
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import pickle
import os
import base64
import email
import json
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from loguru import logger

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
TOKEN_PATH = 'token/gmail_token.pickle'

# Constants for GPT-4o-mini optimization
MAX_EMAILS_PER_BATCH = 20  # Increased due to larger context window
MAX_CONTENT_LENGTH = 120000  # Characters per batch to stay within token limits
DEFAULT_MAX_EMAILS = 10  # Reduced default email limit to manage context length

EMAIL_BATCH_ANALYSIS_PROMPT = """You are a highly capable email analyst using o1-mini. Analyze the following batch of emails based on the user's query and provide a simplified summary.

User Query: {query}
Number of Emails in this Batch: {email_count}
Batch {batch_num} of {total_batches}

Emails:
{email_details}

Provide a simplified analysis that includes:
1. Overview: Brief summary of the email collection matching the query
2. Key Actions:
   - Required actions, prioritized by importance
3. Important Dates:
   - Important dates and deadlines, sorted by urgency

Focus on providing insights that are:
- Concise and actionable
- Organized by importance
- Highlighting only critical information

Respond in JSON format:
{{
    "analysis": {{
        "overview": "brief summary of findings",
        "key_actions": [
            {{"action": "action description", "priority": "high/medium/low", "deadline": "date or timeframe"}}
        ],
        "important_dates": [
            {{"date": "YYYY-MM-DD", "event": "description", "urgency": "high/medium/low"}}
        ]
    }}
}}"""

MERGE_ANALYSES_PROMPT = """You are a highly capable email analysis synthesizer using o1-mini. Merge and synthesize the following email analyses into a single comprehensive summary.

Total Emails Analyzed: {total_emails}
Number of Batches: {num_batches}

Individual Batch Analyses:
{batch_analyses}

Create a unified analysis that:
1. Synthesizes insights across all batches
2. Identifies complex patterns and relationships
3. Eliminates redundancies while preserving unique insights
4. Prioritizes findings based on importance and urgency
5. Provides strategic recommendations based on the complete dataset

Use the same detailed JSON format as the individual analyses, but ensure the merged analysis:
- Captures the full scope of communications
- Highlights evolving patterns across time periods
- Identifies interconnected themes and relationships
- Provides comprehensive action items and recommendations

The merged analysis should be more insightful than the sum of individual analyses."""

# Add new prompt for email drafting
EMAIL_DRAFT_PROMPT = """You are a professional email composer. Draft an email based on the following requirements:

Purpose: {purpose}
To: {recipient}
Additional Context: {context}

Guidelines for composition:
- Write in a professional yet friendly tone
- Be concise and clear
- Use appropriate greeting and closing
- Maintain proper email etiquette
- Highlight key points or requests clearly

Respond in JSON format:
{{
    "draft": {{
        "subject": "proposed email subject",
        "body": "complete email body with greeting and signature",
        "tone": "brief description of the tone used",
        "key_points": ["main points covered in the email"]
    }}
}}"""

class EmailAgent(BaseAgent):
    """Agent responsible for handling email operations using Gmail API"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.service = None
        self.llm = ChatOpenAI(
            model_name=config.get("model_name", "o1-mini"),
            temperature=config.get("temperature", 0),
            api_key=config["openai_api_key"]
        )
        self.batch_analysis_prompt = ChatPromptTemplate.from_template(EMAIL_BATCH_ANALYSIS_PROMPT)
        self.merge_prompt = ChatPromptTemplate.from_template(MERGE_ANALYSES_PROMPT)
        self._drafts = {}
        self._latest_draft_id = None
        self._authenticate()

    def _chunk_emails(self, emails: List[Dict[str, str]], max_chars: int = MAX_CONTENT_LENGTH) -> List[List[Dict[str, str]]]:
        """Split emails into chunks based on content length rather than fixed count"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for email in emails:
            # Calculate approximate length of this email
            email_length = len(str(email))
            
            # If adding this email would exceed max_chars, start a new chunk
            if current_length + email_length > max_chars and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_length = 0
            
            current_chunk.append(email)
            current_length += email_length
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def _truncate_email_content(self, content: str, max_length: int = 5000) -> str:
        """Intelligently truncate email content while preserving important information"""
        if len(content) <= max_length:
            return content
        
        # Try to find a good breaking point
        break_point = content.rfind('\n', 0, max_length)
        if break_point == -1:
            break_point = content.rfind('. ', 0, max_length)
        if break_point == -1:
            break_point = max_length
        
        return content[:break_point] + "\n[Content truncated...]"

    async def _analyze_email_batch(self, emails: List[Dict[str, str]], query: str, batch_num: int, total_batches: int) -> Dict[str, Any]:
        """Analyze a single batch of emails with simplified content handling"""
        email_details = []
        for email_data in emails:
            # Truncate content if necessary while preserving important parts
            truncated_content = self._truncate_email_content(email_data['content'])
            email_details.append(
                f"From: {email_data['from']}\n"
                f"Subject: {email_data['subject']}\n"
                f"Date: {email_data['date']}\n"
                f"Content: {truncated_content}\n"
                f"---"
            )

        chain = self.batch_analysis_prompt | self.llm
        result = await chain.ainvoke({
            "query": query,
            "email_count": len(emails),
            "batch_num": batch_num,
            "total_batches": total_batches,
            "email_details": "\n\n".join(email_details)
        })

        return json.loads(result.content)

    async def _analyze_emails_by_query(self, query: str, days_back: int = 30, max_emails: int = DEFAULT_MAX_EMAILS) -> AgentResponse:
        """Analyze emails matching the query with optimized batch processing"""
        try:
            gmail_query = self._build_gmail_query(query, days_back)
            
            results = self.service.users().messages().list(
                userId='me',
                q=gmail_query,
                maxResults=max_emails
            ).execute()
            
            messages = results.get('messages', [])
            if not messages:
                return AgentResponse(
                    success=True,
                    message="No emails found matching the criteria",
                    data={"analysis": {
                        "overview": "No emails found matching the query",
                        "key_findings": {
                            "main_themes": [],
                            "important_dates": [],
                            "required_actions": [],
                            "risks_opportunities": []
                        },
                        "sender_analysis": [],
                        "timeline_insights": {
                            "progression": "No timeline available",
                            "critical_dates": [],
                            "patterns": []
                        },
                        "content_analysis": {
                            "key_points": [],
                            "unresolved_items": [],
                            "important_references": []
                        },
                        "recommended_actions": []
                    }}
                )

            # Get content for all matching emails
            all_emails = []
            for msg in messages:
                email_data = self._get_email_content(msg['id'])
                all_emails.append(email_data)

            # Split emails into optimized chunks
            chunks = self._chunk_emails(all_emails)
            batch_analyses = []
            
            for i, chunk in enumerate(chunks, 1):
                analysis = await self._analyze_email_batch(chunk, query, i, len(chunks))
                batch_analyses.append(analysis)

            # Merge analyses if there are multiple batches
            final_analysis = await self._merge_analyses(batch_analyses, len(messages))
            
            return AgentResponse(
                success=True,
                message=f"Analyzed {len(messages)} emails matching your criteria",
                data=final_analysis
            )
            
        except Exception as e:
            logger.error(f"Error analyzing emails: {str(e)}")
            return AgentResponse(
                success=False,
                message=f"Error analyzing emails: {str(e)}",
                data=None
            )

    async def _merge_analyses(self, analyses: List[Dict[str, Any]], total_emails: int) -> Dict[str, Any]:
        """Merge multiple batch analyses into a single comprehensive analysis"""
        if len(analyses) == 1:
            return analyses[0]

        chain = self.merge_prompt | self.llm
        result = await chain.ainvoke({
            "total_emails": total_emails,
            "num_batches": len(analyses),
            "batch_analyses": json.dumps(analyses, indent=2)
        })

        return json.loads(result.content)

    def _authenticate(self):
        """Authenticate with Gmail API"""
        creds = None
        os.makedirs('token', exist_ok=True)
        
        if os.path.exists(TOKEN_PATH):
            with open(TOKEN_PATH, 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_config(
                    {
                        "installed": {
                            "client_id": self.config["google_client_id"],
                            "client_secret": self.config["google_client_secret"],
                            "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob"],
                            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                            "token_uri": "https://oauth2.googleapis.com/token"
                        }
                    },
                    SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            with open(TOKEN_PATH, 'wb') as token:
                pickle.dump(creds, token)
        
        self.service = build('gmail', 'v1', credentials=creds)

    def _get_email_content(self, msg_id: str) -> Dict[str, str]:
        """Get full email content including body"""
        try:
            message = self.service.users().messages().get(
                userId='me',
                id=msg_id,
                format='full'
            ).execute()
            
            headers = message['payload']['headers']
            from_address = next(h['value'] for h in headers if h['name'].lower() == 'from')
            subject = next(h['value'] for h in headers if h['name'].lower() == 'subject')
            date = next(h['value'] for h in headers if h['name'].lower() == 'date')
            
            # Get email body
            if 'parts' in message['payload']:
                parts = message['payload']['parts']
                content = ""
                for part in parts:
                    if part['mimeType'] == 'text/plain':
                        data = part['body'].get('data', '')
                        if data:
                            content += base64.urlsafe_b64decode(data).decode()
            else:
                data = message['payload']['body'].get('data', '')
                content = base64.urlsafe_b64decode(data).decode() if data else ""
            
            return {
                'from': from_address,
                'subject': subject,
                'date': date,
                'content': content
            }
        except Exception as e:
            logger.error(f"Error getting email content: {str(e)}")
            return {
                'from': 'Error',
                'subject': 'Error',
                'date': 'Error',
                'content': f'Error retrieving content: {str(e)}'
            }

    def _build_gmail_query(self, query: str, days_back: int) -> str:
        """Build Gmail API query string"""
        date_limit = (datetime.now() - timedelta(days=days_back)).strftime('%Y/%m/%d')
        if query:
            return f"after:{date_limit} ({query})"
        return f"after:{date_limit}"

    async def _send_email(self, to: str, subject: str, body: str) -> AgentResponse:
        """Send an email"""
        if not all([to, subject, body]):
            return AgentResponse(
                success=False,
                message="Missing required parameters (to, subject, body)",
                data=None
            )
        
        try:
            # Create message container with proper email formatting
            message = MIMEText(body, 'plain', 'utf-8')
            sender = self.service.users().getProfile(userId='me').execute()['emailAddress']
            
            # Format headers according to RFC 2822
            message['From'] = sender
            message['To'] = to.strip()  # Ensure no whitespace
            message['Subject'] = subject
            
            # Convert the message to a string first
            raw_message = message.as_string()
            
            # Then encode it
            raw = base64.urlsafe_b64encode(raw_message.encode('utf-8')).decode('utf-8')
            
            # Send the email
            self.service.users().messages().send(
                userId='me',
                body={'raw': raw}
            ).execute()
            
            return AgentResponse(
                success=True,
                message=f"Email sent successfully to {to}",
                data=None
            )
            
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            return AgentResponse(
                success=False,
                message=f"Error sending email: {str(e)}",
                data=None
            )

    async def _send_latest_draft(self) -> AgentResponse:
        """Send the most recently created draft"""
        if not self._latest_draft_id or self._latest_draft_id not in self._drafts:
            return AgentResponse(
                success=False,
                message="No recent draft found to send. Please create a new draft first.",
                data=None
            )
        
        draft = self._drafts[self._latest_draft_id]
        result = await self._send_email(
            to=draft["to"],
            subject=draft["subject"],
            body=draft["body"]
        )
        
        if result.success:
            # Remove the draft after successful sending
            del self._drafts[self._latest_draft_id]
            self._latest_draft_id = None
        
        return result

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate the input data"""
        return (
            isinstance(input_data, dict) and
            "action" in input_data and
            isinstance(input_data.get("parameters", {}), dict)
        ) 

    async def _draft_email(self, to: str, purpose: str, context: str = "") -> AgentResponse:
        """Draft an email using LLM for review"""
        try:
            chain = ChatPromptTemplate.from_template(EMAIL_DRAFT_PROMPT) | self.llm
            result = await chain.ainvoke({
                "purpose": purpose,
                "recipient": to,
                "context": context
            })
            
            draft = json.loads(result.content)
            draft_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save draft in memory
            self._drafts[draft_id] = {
                "to": to,
                "subject": draft["draft"]["subject"],
                "body": draft["draft"]["body"],
                "created_at": datetime.now().isoformat()
            }
            self._latest_draft_id = draft_id
            
            return AgentResponse(
                success=True,
                message="Email draft created for review",
                data={
                    "draft": draft["draft"],
                    "to": to,
                    "requires_review": True,
                    "instructions": "To send this email, simply confirm by saying 'yes', 'send it', or 'looks good'"
                }
            )
        except Exception as e:
            logger.error(f"Error drafting email: {str(e)}")
            return AgentResponse(
                success=False,
                message=f"Error drafting email: {str(e)}",
                data=None
            )

    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """Process email-related requests"""
        if not self.validate_input(input_data):
            return AgentResponse(
                success=False,
                message="Invalid input data format",
                data=None
            )
        
        action = input_data.get("action")
        params = input_data.get("parameters", {})
        
        try:
            if action == "summarize_inbox":
                return await self._analyze_emails_by_query(
                    query=params.get("query", ""),
                    days_back=params.get("days_back", 30),
                    max_emails=params.get("max_emails", DEFAULT_MAX_EMAILS)
                )
            elif action == "draft_email":
                return await self._draft_email(
                    to=params.get("to"),
                    purpose=params.get("purpose", ""),
                    context=params.get("context", "")
                )
            elif action == "confirm_send":
                return await self._send_latest_draft()
            elif action == "send_email":
                if not params.get("reviewed", False):
                    return AgentResponse(
                        success=False,
                        message="Email must be reviewed before sending",
                        data=None
                    )
                return await self._send_email(
                    to=params.get("to"),
                    subject=params.get("subject"),
                    body=params.get("body")
                )
            else:
                return AgentResponse(
                    success=False,
                    message=f"Unknown action: {action}",
                    data=None
                )
        
        except Exception as e:
            logger.error(f"Error processing email request: {str(e)}")
            return AgentResponse(
                success=False,
                message=f"Error processing email request: {str(e)}",
                data=None
            ) 