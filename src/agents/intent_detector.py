from typing import Dict, Any
from .base import BaseAgent, AgentResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
import json

INTENT_PROMPT = """You are an intent detection agent for a personal assistant.
Given the user input, determine which agent should handle the request and extract relevant information.

Available agents and their capabilities:

1. Email Agent (agent: "email"):
   - summarize_inbox: Analyze emails matching specific criteria
     Parameters:
     - query: (optional) search terms or criteria (e.g., "job interviews", "from amazon")
     - days_back: (optional) number of days to look back, default 30
     - max_emails: (optional) maximum number of emails to analyze, default 10
     The analysis provides:
     - Overview of matching emails
     - Key actions needed
     - Important dates
   
   - draft_email: Create an email draft for review
     Parameters:
     - to: recipient email address
     - purpose: the purpose or intent of the email
     - context: (optional) additional context or specific points to include
   
   - confirm_send: Send the most recently created draft
     This action should be detected when:
     - User expresses approval or confirmation of the draft
     - User indicates they want to proceed with sending
     - User shows satisfaction with the draft content
     Parameters: none (uses the latest draft)
   
   - send_email: Send a reviewed email
     Parameters:
     - to: recipient email address
     - subject: email subject
     - body: email content
     - reviewed: must be true to send

2. Calendar Agent (agent: "calendar"):
   - list_events: List upcoming calendar events
     Parameters:
     - days: (optional) number of days to look ahead, default 7
   
   - create_event: Create or modify a calendar event
     Parameters:
     - summary: event title
     - start_time: start time (e.g., "Thursday 4pm this week")
     - end_time: (optional) end time
     - description: (optional) event description
     - attendees: (optional) list of attendee email addresses
     - duration_minutes: (optional) event duration in minutes, default 30
     - is_modification: (boolean) whether this is modifying an existing draft
     - modifications: (optional) specific changes to make to the current draft
     - confirmed: (boolean) whether to create the event or just show draft

User input: {user_input}

Consider the context:
1. For initial event creation requests (e.g., "schedule a meeting", "create an event"), ALWAYS set confirmed=false
   to show the draft first.
2. If a calendar event draft was just shown and the user wants to modify it (e.g., change duration, time, description),
   treat it as a modification to the existing draft rather than creating a new event.
3. If the user expresses confirmation or approval in ANY way (e.g., "looks good", "confirmed", "yes", "that's correct", 
   "perfect", "go ahead", "create it", "schedule it", etc.), set confirmed=true in the parameters.
4. If the user wants to modify something, identify what they want to change and include it in the modifications parameter.
5. For calendar events, always interpret day references (e.g., "Thursday") relative to the current week unless explicitly 
   stated otherwise.
6. For duration modifications:
   - When user says "extend to X minutes/hours", set the absolute duration
   - When user says "extend by X minutes/hours", use relative duration with "+="
   - Examples:
     - "extend to 1 hour" -> duration_minutes: 60
     - "extend by 30 minutes" -> duration_minutes: "+=30"
     - "make it an hour longer" -> duration_minutes: "+=60"

Example responses:

For initial event creation:
{{"agent": "calendar", "action": "create_event", "parameters": {{"summary": "New Event", "start_time": "Thursday 4pm this week", "duration_minutes": 30, "confirmed": false}}}}

For modifying duration (absolute):
{{"agent": "calendar", "action": "create_event", "parameters": {{"is_modification": true, "modifications": {{"duration_minutes": 60}}}}}}

For modifying duration (relative):
{{"agent": "calendar", "action": "create_event", "parameters": {{"is_modification": true, "modifications": {{"duration_minutes": "+=30"}}}}}}

For confirming a draft:
{{"agent": "calendar", "action": "create_event", "parameters": {{"confirmed": true}}}}"""

class IntentDetectorAgent(BaseAgent):
    """Agent responsible for detecting user intent and routing to appropriate agent"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm = ChatOpenAI(
            model_name=config.get("model_name", "o1-mini"),
            temperature=config.get("temperature", 0),
            api_key=config["openai_api_key"]
        )
        self.prompt = ChatPromptTemplate.from_template(INTENT_PROMPT)
        self._last_draft = None  # Store the last draft event details
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """Process user input and detect intent"""
        if not self.validate_input(input_data):
            return AgentResponse(
                success=False,
                message="Invalid input data format",
                data=None
            )
        
        try:
            # Check if this is a confirmation for a draft event
            user_input = input_data["user_input"].lower().strip()
            confirmation_phrases = [
                "looks good", "confirmed", "yes", "that's correct", "perfect",
                "go ahead", "create it", "schedule it", "that works", "proceed",
                "ok", "good", "fine", "alright", "sure"
            ]
            
            # If we have a draft and user input matches confirmation phrases
            if any(phrase in user_input for phrase in confirmation_phrases):
                return AgentResponse(
                    success=True,
                    message="Intent detected successfully",
                    data={
                        "raw_response": json.dumps({
                            "agent": "calendar",
                            "action": "create_event",
                            "parameters": {
                                "confirmed": True
                            }
                        })
                    }
                )
            
            # Use LLM for all other intent detection
            chain = self.prompt | self.llm
            result = await chain.ainvoke({"user_input": input_data["user_input"]})
            
            # Clean up the response
            response_text = result.content.strip()
            logger.debug(f"Raw LLM response: {response_text}")
            
            # Remove any markdown code block formatting if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            logger.debug(f"Cleaned LLM response: {response_text}")
            
            # Validate the response format
            try:
                parsed = json.loads(response_text)
                required_fields = ["agent", "action", "parameters"]
                for field in required_fields:
                    if field not in parsed:
                        raise KeyError(f"Missing required field: {field}")
                
                # Return the cleaned response
                return AgentResponse(
                    success=True,
                    message="Intent detected successfully",
                    data={"raw_response": response_text}
                )
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Invalid LLM response format: {str(e)}")
                logger.error(f"Response was: {response_text}")
                return AgentResponse(
                    success=False,
                    message="Failed to parse intent. Please try rephrasing your request.",
                    data=None
                )
            
        except Exception as e:
            logger.error(f"Error detecting intent: {str(e)}")
            return AgentResponse(
                success=False,
                message=f"Error detecting intent: {str(e)}",
                data=None
            )
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate the input data"""
        return isinstance(input_data, dict) and "user_input" in input_data 