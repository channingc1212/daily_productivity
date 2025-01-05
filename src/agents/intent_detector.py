from typing import Dict, Any
from .base import BaseAgent, AgentResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

INTENT_PROMPT = """You are an intent detection agent for a personal assistant.
Given the user input, determine which agent should handle the request and extract relevant information.

Available agents and their capabilities:

1. Email Agent (agent: "email"):
   - summarize_inbox: Get a summary of recent emails
     Parameters:
     - max_emails: (optional) number of emails to summarize, default 5
   
   - send_email: Send a new email
     Parameters:
     - to: recipient email address
     - subject: email subject
     - body: email content

2. Calendar Agent (agent: "calendar"):
   - list_events: List upcoming calendar events
     Parameters:
     - days: (optional) number of days to look ahead, default 7
   
   - create_event: Create a new calendar event
     Parameters:
     - summary: event title
     - start_time: start time in ISO format (e.g., "2024-01-05T15:00:00Z")
     - end_time: end time in ISO format
     - description: (optional) event description
     - attendees: (optional) list of attendee email addresses

User input: {user_input}

Respond with a valid JSON object containing:
- agent: either "email" or "calendar"
- action: the specific action to perform
- parameters: a dictionary of parameters for the action

Example responses:

For "Show me my recent emails":
{{"agent": "email", "action": "summarize_inbox", "parameters": {{"max_emails": 5}}}}

For "Schedule a meeting with john@example.com tomorrow at 2pm for 1 hour":
{{"agent": "calendar", "action": "create_event", "parameters": {{"summary": "Meeting", "start_time": "2024-01-06T14:00:00Z", "end_time": "2024-01-06T15:00:00Z", "attendees": ["john@example.com"]}}}}"""

class IntentDetectorAgent(BaseAgent):
    """Agent responsible for detecting user intent and routing to appropriate agent"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm = ChatOpenAI(
            model_name=config.get("model_name", "gpt-3.5-turbo"),
            temperature=config.get("temperature", 0),
            api_key=config["openai_api_key"]
        )
        self.prompt = ChatPromptTemplate.from_template(INTENT_PROMPT)
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """Process user input and detect intent"""
        if not self.validate_input(input_data):
            return AgentResponse(
                success=False,
                message="Invalid input data format",
                data=None
            )
        
        try:
            chain = self.prompt | self.llm
            result = await chain.ainvoke({"user_input": input_data["user_input"]})
            
            # Parse the response and return structured intent
            response_text = result.content
            
            return AgentResponse(
                success=True,
                message="Intent detected successfully",
                data={"raw_response": response_text}
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