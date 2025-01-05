from typing import Dict, Any
from .base import BaseAgent, AgentResponse
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pickle
import os
from datetime import datetime, timedelta
from loguru import logger

SCOPES = ['https://www.googleapis.com/auth/calendar']
TOKEN_PATH = 'token/calendar_token.pickle'

class CalendarAgent(BaseAgent):
    """Agent responsible for handling calendar operations using Google Calendar API"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Calendar API"""
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
        
        self.service = build('calendar', 'v3', credentials=creds)
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """Process calendar-related requests"""
        if not self.validate_input(input_data):
            return AgentResponse(
                success=False,
                message="Invalid input data format",
                data=None
            )
        
        action = input_data.get("action")
        params = input_data.get("parameters", {})
        
        try:
            if action == "list_events":
                return await self._list_events(
                    days=params.get("days", 7)
                )
            elif action == "create_event":
                return await self._create_event(
                    summary=params.get("summary"),
                    start_time=params.get("start_time"),
                    end_time=params.get("end_time"),
                    description=params.get("description", ""),
                    attendees=params.get("attendees", [])
                )
            else:
                return AgentResponse(
                    success=False,
                    message=f"Unknown action: {action}",
                    data=None
                )
        
        except Exception as e:
            logger.error(f"Error processing calendar request: {str(e)}")
            return AgentResponse(
                success=False,
                message=f"Error processing calendar request: {str(e)}",
                data=None
            )
    
    async def _list_events(self, days: int = 7) -> AgentResponse:
        """List upcoming events"""
        try:
            now = datetime.utcnow()
            time_max = (now + timedelta(days=days)).isoformat() + 'Z'
            
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=now.isoformat() + 'Z',
                timeMax=time_max,
                maxResults=10,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            event_list = []
            
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                event_list.append({
                    'summary': event.get('summary', 'No title'),
                    'start': start,
                    'description': event.get('description', ''),
                    'attendees': [
                        attendee['email']
                        for attendee in event.get('attendees', [])
                    ]
                })
            
            return AgentResponse(
                success=True,
                message=f"Retrieved {len(event_list)} events",
                data={"events": event_list}
            )
            
        except Exception as e:
            logger.error(f"Error listing events: {str(e)}")
            return AgentResponse(
                success=False,
                message=f"Error listing events: {str(e)}",
                data=None
            )
    
    async def _create_event(
        self,
        summary: str,
        start_time: str,
        end_time: str,
        description: str = "",
        attendees: list = None
    ) -> AgentResponse:
        """Create a calendar event"""
        if not all([summary, start_time, end_time]):
            return AgentResponse(
                success=False,
                message="Missing required parameters (summary, start_time, end_time)",
                data=None
            )
        
        try:
            event = {
                'summary': summary,
                'description': description,
                'start': {
                    'dateTime': start_time,
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': end_time,
                    'timeZone': 'UTC',
                },
            }
            
            if attendees:
                event['attendees'] = [{'email': email} for email in attendees]
            
            event = self.service.events().insert(
                calendarId='primary',
                body=event,
                sendUpdates='all'
            ).execute()
            
            return AgentResponse(
                success=True,
                message="Event created successfully",
                data={"event_id": event['id']}
            )
            
        except Exception as e:
            logger.error(f"Error creating event: {str(e)}")
            return AgentResponse(
                success=False,
                message=f"Error creating event: {str(e)}",
                data=None
            )
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate the input data"""
        return (
            isinstance(input_data, dict) and
            "action" in input_data and
            isinstance(input_data.get("parameters", {}), dict)
        ) 