from typing import Dict, Any
from .base import BaseAgent, AgentResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
import json

INTENT_PROMPT = """You are an intent detection system for a personal assistant. Based on the user's input and current conversation state, determine the appropriate agent and action to handle the request.

Current conversation state: {conversation_state}

User input: {user_input}

Detect the intent and return a JSON response with the following structure:
{{
    "agent": "email|calendar",
    "action": "specific_action",
    "parameters": {{
        // For calendar events:
        "summary": "event title",
        "start_time": "YYYY-MM-DD HH:mm",  // Format time in 24-hour format with date
        "duration_minutes": 30,  // Default duration
        "description": "optional description",
        "attendees": [],  // Optional list of attendees
        // For modifications:
        "modification": {{
            "type": "duration|time|description",
            "value": "the new value"
        }}
    }},
    "requires_confirmation": true|false,
    "context": {{
        "is_modification": true|false,
        "modification_type": "duration|time|description|etc",
        "original_draft_preserved": true|false
    }}
}}

For calendar events:
- Extract time expressions from natural language:
  * Absolute times: "11am" -> "HH:mm"
  * Relative dates: "tomorrow", "next week", "this Friday"
  * Combined expressions: "tomorrow at 11am" -> "YYYY-MM-DD 11:00"
  * Time ranges: "from 2pm to 4pm", "for 2 hours"
- Always include the full date and time in start_time
- Default duration is 30 minutes unless specified

For confirmations:
- When user expresses agreement (e.g., "looks good", "yes", "that works", "confirmed", "okay", "sure", "perfect", "great"):
  * Set action to "confirm_draft"
  * No need for modification parameters
  * Set requires_confirmation to false

Examples:
1. "create an event tomorrow 11am for shopping" ->
{{
    "agent": "calendar",
    "action": "create_event",
    "parameters": {{
        "summary": "Shopping",
        "start_time": "2025-01-09 11:00",
        "duration_minutes": 30,
        "description": "Shopping event"
    }},
    "requires_confirmation": true,
    "context": {{
        "is_modification": false,
        "original_draft_preserved": false
    }}
}}

2. "extend to 1 hour" ->
{{
    "agent": "calendar",
    "action": "modify_draft",
    "parameters": {{
        "modification": {{
            "type": "duration",
            "value": "60"
        }}
    }},
    "requires_confirmation": true,
    "context": {{
        "is_modification": true,
        "modification_type": "duration",
        "original_draft_preserved": true
    }}
}}

3. "looks good" ->
{{
    "agent": "calendar",
    "action": "confirm_draft",
    "parameters": {{}},
    "requires_confirmation": false,
    "context": {{
        "is_modification": false,
        "original_draft_preserved": true
    }}
}}
"""

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
        self.conversation_history = []
        self._current_state = {
            "current_draft": None,
            "last_action": None,
            "pending_confirmation": False
        }
    
    def _update_state(self, intent_response: Dict[str, Any]) -> None:
        """Update the conversation state based on the detected intent"""
        try:
            self._current_state["last_action"] = {
                "agent": intent_response.get("agent"),
                "action": intent_response.get("action"),
                "context": intent_response.get("context", {})
            }
            self._current_state["pending_confirmation"] = intent_response.get("requires_confirmation", False)
        except Exception as e:
            logger.error(f"Error updating state: {str(e)}")
    
    def _get_conversation_state(self) -> str:
        """Generate a description of the current conversation state"""
        try:
            if not self._current_state.get("last_action"):
                return "No previous context. Starting fresh."
            
            state_desc = []
            if self._current_state.get("current_draft"):
                state_desc.append("There is a draft waiting for confirmation or modification.")
            
            if self._current_state.get("last_action"):
                action = self._current_state["last_action"]
                state_desc.append(f"Last action: {action.get('agent')} agent performed {action.get('action')}")
            
            if self._current_state.get("pending_confirmation"):
                state_desc.append("Waiting for user confirmation.")
            
            return " ".join(state_desc) if state_desc else "No active context."
        except Exception as e:
            logger.error(f"Error getting conversation state: {str(e)}")
            return "Error retrieving conversation state."
    
    def _clean_llm_response(self, response_text: str) -> str:
        """Clean and validate LLM response text"""
        try:
            # Remove any markdown formatting
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                parts = cleaned.split("```")
                if len(parts) >= 2:
                    cleaned = parts[1]
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:]
            cleaned = cleaned.strip()
            
            # Parse JSON to validate structure
            intent_data = json.loads(cleaned)
            
            # Validate time format for calendar events
            if (
                intent_data.get("agent") == "calendar" and
                intent_data.get("action") == "create_event" and
                "parameters" in intent_data
            ):
                params = intent_data["parameters"]
                if "start_time" in params:
                    # Ensure time is in YYYY-MM-DD HH:mm format
                    start_time = params["start_time"]
                    try:
                        from datetime import datetime
                        datetime.strptime(start_time, "%Y-%m-%d %H:%M")
                    except ValueError:
                        logger.error(f"Invalid time format: {start_time}")
                        raise ValueError("Time must be in YYYY-MM-DD HH:mm format")
            
            return cleaned
        except Exception as e:
            logger.error(f"Error cleaning LLM response: {str(e)}")
            raise
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """Process user input and detect intent"""
        if not self.validate_input(input_data):
            return AgentResponse(
                success=False,
                message="Invalid input data format",
                data=None
            )
        
        try:
            # Get current conversation state
            conversation_state = self._get_conversation_state()
            
            # Use LLM for intent detection with full context
            chain = self.prompt | self.llm
            result = await chain.ainvoke({
                "user_input": input_data["user_input"],
                "conversation_state": conversation_state
            })
            
            # Clean and validate the response
            try:
                response_text = self._clean_llm_response(result.content)
                intent_response = json.loads(response_text)
                
                # Validate required fields
                required_fields = ["agent", "action", "parameters", "context"]
                if not all(field in intent_response for field in required_fields):
                    raise ValueError("Missing required fields in intent response")
                
                # Special handling for modifications
                if intent_response.get("context", {}).get("is_modification"):
                    if not intent_response.get("parameters", {}).get("modification"):
                        raise ValueError("Missing modification details in parameters")
                    
                    # Ensure modification has type and value
                    mod = intent_response["parameters"]["modification"]
                    if not all(k in mod for k in ["type", "value"]):
                        raise ValueError("Modification must specify type and value")
                
                # Update conversation state
                self._update_state(intent_response)
                
                # Store the raw response for the agent
                return AgentResponse(
                    success=True,
                    message="Intent detected successfully",
                    data={"raw_response": response_text}
                )
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON format in LLM response: {str(e)}")
                return AgentResponse(
                    success=False,
                    message="Failed to parse intent. Please try rephrasing your request.",
                    data=None
                )
            except ValueError as e:
                logger.error(f"Invalid intent response format: {str(e)}")
                return AgentResponse(
                    success=False,
                    message="Invalid intent format. Please try rephrasing your request.",
                    data=None
                )
            
        except Exception as e:
            logger.error(f"Error detecting intent: {str(e)}")
            return AgentResponse(
                success=False,
                message="Sorry, I encountered an error. Please try again.",
                data=None
            )
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate the input data"""
        try:
            return (
                isinstance(input_data, dict) and
                "user_input" in input_data and
                isinstance(input_data["user_input"], str) and
                len(input_data["user_input"].strip()) > 0
            )
        except Exception as e:
            logger.error(f"Error validating input: {str(e)}")
            return False 