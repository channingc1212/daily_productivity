from typing import Dict, Any
from .base import BaseAgent, AgentResponse
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pickle
import os
from loguru import logger

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
TOKEN_PATH = 'token/gmail_token.pickle'

class EmailAgent(BaseAgent):
    """Agent responsible for handling email operations using Gmail API"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.service = None
        self._authenticate()
    
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
                return await self._summarize_inbox(params.get("max_emails", 5))
            elif action == "send_email":
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
    
    async def _summarize_inbox(self, max_emails: int = 5) -> AgentResponse:
        """Summarize recent emails in inbox"""
        try:
            results = self.service.users().messages().list(
                userId='me',
                labelIds=['INBOX'],
                maxResults=max_emails
            ).execute()
            
            messages = results.get('messages', [])
            summaries = []
            
            for msg in messages:
                email = self.service.users().messages().get(
                    userId='me',
                    id=msg['id'],
                    format='metadata',
                    metadataHeaders=['From', 'Subject', 'Date']
                ).execute()
                
                headers = email['payload']['headers']
                summary = {
                    'from': next(h['value'] for h in headers if h['name'] == 'From'),
                    'subject': next(h['value'] for h in headers if h['name'] == 'Subject'),
                    'date': next(h['value'] for h in headers if h['name'] == 'Date')
                }
                summaries.append(summary)
            
            return AgentResponse(
                success=True,
                message=f"Retrieved {len(summaries)} emails",
                data={"emails": summaries}
            )
            
        except Exception as e:
            logger.error(f"Error summarizing inbox: {str(e)}")
            return AgentResponse(
                success=False,
                message=f"Error summarizing inbox: {str(e)}",
                data=None
            )
    
    async def _send_email(self, to: str, subject: str, body: str) -> AgentResponse:
        """Send an email"""
        if not all([to, subject, body]):
            return AgentResponse(
                success=False,
                message="Missing required parameters (to, subject, body)",
                data=None
            )
        
        try:
            import base64
            from email.mime.text import MIMEText
            
            message = MIMEText(body)
            message['to'] = to
            message['subject'] = subject
            
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            
            self.service.users().messages().send(
                userId='me',
                body={'raw': raw}
            ).execute()
            
            return AgentResponse(
                success=True,
                message="Email sent successfully",
                data=None
            )
            
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            return AgentResponse(
                success=False,
                message=f"Error sending email: {str(e)}",
                data=None
            )
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate the input data"""
        return (
            isinstance(input_data, dict) and
            "action" in input_data and
            isinstance(input_data.get("parameters", {}), dict)
        ) 