from typing import Dict, Any, List, Optional
from .base import BaseAgent, AgentResponse
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import pickle
import os
import json
from datetime import datetime, timedelta
import pytz
from loguru import logger

SCOPES = ['https://www.googleapis.com/auth/calendar']
TOKEN_PATH = 'token/calendar_token.pickle'

CALENDAR_SUMMARY_PROMPT = """You are a calendar analyst using o1-mini. Analyze the following calendar events and provide a high-level summary.

Events:
{events}

Provide a concise summary that includes:
1. Overview: Brief summary of the schedule
2. Key Events: Important meetings or deadlines
3. Time Blocks: Major time commitments
4. Free Time: Notable gaps in the schedule

Focus on providing insights that are:
- Organized chronologically
- Highlighting important commitments
- Identifying scheduling patterns

Respond in JSON format:
{
    "summary": {
        "overview": "brief overview of the schedule",
        "key_events": [
            {"time": "datetime", "event": "description", "importance": "high/medium/low"}
        ],
        "time_blocks": [
            {"period": "morning/afternoon/evening", "status": "busy/moderate/free", "description": "brief description"}
        ],
        "scheduling_notes": ["relevant notes about the schedule"]
    }
}"""

DATE_PARSING_PROMPT = """You are a date and time parser. Parse the following natural language date/time expression into a specific date and time.

Current date and time for reference: {current_time}
Timezone: {timezone}
Expression to parse: {time_str}

Instructions:
1. Use the current date and time as reference point
2. For relative expressions like "this week", use the current week
3. For day names (e.g., "Thursday"), find the next occurrence from current date
4. Return ONLY a JSON object with the parsed date/time components

Example response format:
{{
    "parsed_datetime": {{
        "year": 2025,
        "month": 1,
        "day": 9,
        "hour": 16,
        "minute": 0,
        "is_ambiguous": false
    }}
}}

Return only the JSON object, no other text or formatting."""

class CalendarAgent(BaseAgent):
    """Agent responsible for handling calendar operations using Google Calendar API"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.service = None
        # Use system timezone
        self.timezone = datetime.now().astimezone().tzinfo
        self.llm = ChatOpenAI(
            model_name=config.get("model_name", "o1-mini"),
            temperature=config.get("temperature", 0),
            api_key=config["openai_api_key"]
        )
        self.summary_prompt = ChatPromptTemplate.from_template(CALENDAR_SUMMARY_PROMPT)
        self.date_prompt = ChatPromptTemplate.from_template(DATE_PARSING_PROMPT)
        self._current_draft = None  # Store the current draft event
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
    
    async def _parse_time(self, time_str: str, default_duration: int = 30) -> tuple[datetime, datetime]:
        """Parse time string using LLM and return start and end times"""
        try:
            # Get current time in the configured timezone
            current_time = datetime.now(self.timezone)
            
            # Call LLM to parse the date/time
            chain = self.date_prompt | self.llm
            result = await chain.ainvoke({
                "time_str": time_str,
                "current_time": current_time.strftime("%Y-%m-%d %H:%M %Z"),
                "timezone": str(self.timezone)
            })
            
            try:
                # Parse the LLM response
                response_text = result.content.strip()
                logger.debug(f"Raw LLM response: {response_text}")
                
                # Remove any markdown code block formatting if present
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                response_text = response_text.strip()
                
                logger.debug(f"Cleaned LLM response: {response_text}")
                
                # Try to parse the JSON response
                parsed = json.loads(response_text)
                parsed_dt = parsed.get("parsed_datetime", {})
                
                if not parsed_dt:
                    logger.error("No parsed_datetime in LLM response")
                    return None, None
                
                # Create datetime object from parsed components with explicit type conversion
                try:
                    start_time = datetime(
                        year=int(parsed_dt.get("year", current_time.year)),
                        month=int(parsed_dt.get("month", current_time.month)),
                        day=int(parsed_dt.get("day", current_time.day)),
                        hour=int(parsed_dt.get("hour", current_time.hour)),
                        minute=int(parsed_dt.get("minute", 0)),
                        tzinfo=self.timezone
                    )
                    
                    # For day-of-week references (e.g., "Thursday"), calculate the correct date
                    if any(day in time_str.lower() for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
                        # Get the target day of week (0 = Monday, 6 = Sunday)
                        day_map = {
                            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                            "friday": 4, "saturday": 5, "sunday": 6
                        }
                        for day, day_num in day_map.items():
                            if day in time_str.lower():
                                target_day = day_num
                                break
                        else:
                            target_day = start_time.weekday()
                        
                        # Calculate days until next occurrence
                        current_day = current_time.weekday()
                        days_ahead = target_day - current_day
                        if days_ahead <= 0:  # Target day has passed this week
                            days_ahead += 7
                        
                        # Adjust the date while keeping the time
                        start_time = datetime.combine(
                            (current_time + timedelta(days=days_ahead)).date(),
                            start_time.time(),
                            tzinfo=self.timezone
                        )
                    
                    # Calculate end time based on duration
                    end_time = start_time + timedelta(minutes=default_duration)
                    
                    return start_time, end_time
                    
                except (ValueError, TypeError) as e:
                    logger.error(f"Error creating datetime object: {str(e)}")
                    logger.error(f"Parsed datetime components: {parsed_dt}")
                    return None, None
                
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.error(f"Error parsing LLM response: {str(e)}")
                logger.error(f"Raw response: {result.content}")
                return None, None
                
        except Exception as e:
            logger.error(f"Error parsing time: {str(e)}")
            return None, None
    
    def _generate_fallback_analysis(self, events: List[Dict]) -> Dict:
        """Generate a comprehensive fallback analysis when LLM is unavailable"""
        # Group events by day
        events_by_day = {}
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
            day_key = start_dt.strftime('%Y-%m-%d')
            if day_key not in events_by_day:
                events_by_day[day_key] = []
            events_by_day[day_key].append(event)

        # Analyze time blocks
        time_blocks = []
        for day, day_events in events_by_day.items():
            morning_events = [e for e in day_events if 5 <= datetime.fromisoformat(e['start'].get('dateTime', e['start'].get('date')).replace('Z', '+00:00')).hour < 12]
            afternoon_events = [e for e in day_events if 12 <= datetime.fromisoformat(e['start'].get('dateTime', e['start'].get('date')).replace('Z', '+00:00')).hour < 17]
            evening_events = [e for e in day_events if datetime.fromisoformat(e['start'].get('dateTime', e['start'].get('date')).replace('Z', '+00:00')).hour >= 17]
            
            if morning_events:
                time_blocks.append({
                    "period": f"morning ({day})",
                    "status": "busy" if len(morning_events) > 2 else "moderate",
                    "description": f"{len(morning_events)} events: {', '.join(e.get('summary', 'Untitled') for e in morning_events)}"
                })
            if afternoon_events:
                time_blocks.append({
                    "period": f"afternoon ({day})",
                    "status": "busy" if len(afternoon_events) > 2 else "moderate",
                    "description": f"{len(afternoon_events)} events: {', '.join(e.get('summary', 'Untitled') for e in afternoon_events)}"
                })
            if evening_events:
                time_blocks.append({
                    "period": f"evening ({day})",
                    "status": "busy" if len(evening_events) > 2 else "moderate",
                    "description": f"{len(evening_events)} events: {', '.join(e.get('summary', 'Untitled') for e in evening_events)}"
                })

        # Generate overview
        days_with_events = len(events_by_day)
        total_events = len(events)
        busy_days = [day for day, events in events_by_day.items() if len(events) > 2]
        
        overview = f"Your {days_with_events}-day schedule contains {total_events} events"
        if busy_days:
            overview += f", with particularly busy days on {', '.join(busy_days)}"
        
        # Analyze event importance based on various factors
        key_events = []
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            importance = "high"  # Default to high for events with specific attributes
            
            # Determine importance based on various factors
            if not event.get('attendees'):
                importance = "medium"
            if not event.get('description') and not event.get('location'):
                importance = "low"
            
            key_events.append({
                "time": start,
                "event": event.get('summary', 'No title'),
                "importance": importance,
                "location": event.get('location', 'No location'),
                "description": event.get('description', 'No description'),
                "attendees": len(event.get('attendees', []))
            })

        # Generate scheduling notes
        scheduling_notes = []
        if total_events > 0:
            scheduling_notes.append(f"Schedule spans {days_with_events} days with {total_events} total events")
            locations = [e.get('location') for e in events if e.get('location')]
            if locations:
                scheduling_notes.append(f"Events at {len(set(locations))} different locations")
            attendee_events = [e for e in events if e.get('attendees')]
            if attendee_events:
                scheduling_notes.append(f"{len(attendee_events)} events with other attendees")

        return {
            "summary": {
                "overview": overview,
                "key_events": key_events,
                "time_blocks": time_blocks,
                "scheduling_notes": scheduling_notes
            }
        }

    async def _list_events(self, days: int = 7) -> AgentResponse:
        """List and analyze upcoming events"""
        try:
            now = datetime.now(self.timezone)
            time_max = (now + timedelta(days=days)).isoformat()
            
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=now.isoformat(),
                timeMax=time_max,
                maxResults=50,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            if not events:
                return AgentResponse(
                    success=True,
                    message="No upcoming events found",
                    data={"summary": {
                        "overview": "Your schedule is clear",
                        "key_events": [],
                        "time_blocks": [{"period": "all day", "status": "free", "description": "No scheduled events"}],
                        "scheduling_notes": ["Your calendar is completely open"]
                    }}
                )
            
            # Format events for analysis
            event_details = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))
                event_details.append(
                    f"Event: {event.get('summary', 'No title')}\n"
                    f"Time: {start} to {end}\n"
                    f"Location: {event.get('location', 'No location')}\n"
                    f"Description: {event.get('description', 'No description')}\n"
                    f"Attendees: {', '.join(attendee['email'] for attendee in event.get('attendees', []))}\n"
                    f"---"
                )
            
            try:
                # Analyze events using LLM
                chain = self.summary_prompt | self.llm
                result = await chain.ainvoke({
                    "events": "\n\n".join(event_details)
                })
                
                # Try to parse LLM response as JSON
                try:
                    analysis = json.loads(result.content)
                except json.JSONDecodeError:
                    # Fallback to comprehensive analysis if LLM response is not valid JSON
                    logger.warning("Could not parse LLM response as JSON, using fallback analysis")
                    analysis = self._generate_fallback_analysis(events)
                
                return AgentResponse(
                    success=True,
                    message=f"Found {len(events)} events in the next {days} days",
                    data=analysis
                )
                
            except Exception as e:
                # Fallback to comprehensive analysis if LLM fails
                logger.error(f"Error in LLM analysis: {str(e)}")
                return AgentResponse(
                    success=True,
                    message=f"Found {len(events)} events in the next {days} days",
                    data=self._generate_fallback_analysis(events)
                )
            
        except Exception as e:
            logger.error(f"Error listing events: {str(e)}")
            return AgentResponse(
                success=False,
                message=f"Error listing events: {str(e)}",
                data=None
            )
    
    def _modify_draft_event(self, modifications: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Modify the current draft event with the specified changes"""
        if not self._current_draft or not modifications:
            return None
            
        # Create a copy of the current draft
        modified_draft = self._current_draft.copy()
        
        # Handle duration modifications
        if "duration_minutes" in modifications:
            new_duration = modifications["duration_minutes"]
            # Handle relative modifications (e.g., "+=60" for adding an hour)
            if isinstance(new_duration, str) and new_duration.startswith("+="):
                try:
                    additional_minutes = int(new_duration[2:])
                    new_duration = modified_draft["duration_minutes"] + additional_minutes
                except ValueError:
                    return None
            
            # Update duration and end time
            modified_draft["duration_minutes"] = new_duration
            start_time = datetime.fromisoformat(modified_draft["start_time"])
            modified_draft["end_time"] = (start_time + timedelta(minutes=new_duration)).isoformat()
        
        # Handle other modifications
        for key in ["summary", "description", "attendees"]:
            if key in modifications:
                modified_draft[key] = modifications[key]
        
        return modified_draft

    async def _create_event(
        self,
        summary: str = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        description: str = "",
        attendees: list = None,
        duration_minutes: int = 30,
        confirmed: bool = False,
        is_modification: bool = False,
        modifications: Dict[str, Any] = None
    ) -> AgentResponse:
        """Create a calendar event with parameter validation and confirmation"""
        try:
            # If this is a confirmation and we have a draft, use the draft's details
            if confirmed and self._current_draft:
                # Extract details from the current draft
                draft = self._current_draft
                summary = draft.get("summary", summary)
                start_time = draft.get("start_time")
                end_time = draft.get("end_time")
                description = draft.get("description", description)
                attendees = draft.get("attendees", attendees)
                duration_minutes = draft.get("duration_minutes", duration_minutes)
                
                # Create event directly from draft
                event = {
                    'summary': summary,
                    'description': description,
                    'start': {
                        'dateTime': start_time,
                        'timeZone': str(self.timezone),
                    },
                    'end': {
                        'dateTime': end_time,
                        'timeZone': str(self.timezone),
                    },
                }
                
                if attendees:
                    event['attendees'] = [{'email': email} for email in attendees]
                    event['guestsCanModify'] = True
                
                # Insert event and send notifications
                event = self.service.events().insert(
                    calendarId='primary',
                    body=event,
                    sendUpdates='all'
                ).execute()
                
                # Clear the draft after successful creation
                self._current_draft = None
                
                return AgentResponse(
                    success=True,
                    message="Event created successfully",
                    data={
                        "event_id": event['id'],
                        "summary": event['summary'],
                        "start": datetime.fromisoformat(start_time).strftime("%Y-%m-%d %I:%M %p %Z"),
                        "end": datetime.fromisoformat(end_time).strftime("%Y-%m-%d %I:%M %p %Z"),
                        "attendees": attendees
                    }
                )
            
            # Handle modifications to existing draft
            if is_modification and self._current_draft:
                modified_draft = self._modify_draft_event(modifications)
                if modified_draft:
                    # Show the modified draft for confirmation
                    start_time = datetime.fromisoformat(modified_draft["start_time"])
                    end_time = datetime.fromisoformat(modified_draft["end_time"])
                    
                    return AgentResponse(
                        success=True,
                        message="Here's the modified event. Please confirm if this is correct, or let me know what needs to be changed:",
                        data={
                            "draft_event": {
                                "summary": modified_draft.get("summary"),
                                "description": modified_draft.get("description", ""),
                                "start": start_time.strftime("%Y-%m-%d %I:%M %p %Z"),
                                "end": end_time.strftime("%Y-%m-%d %I:%M %p %Z"),
                                "duration": f"{modified_draft['duration_minutes']} minutes",
                                "attendees": modified_draft.get("attendees", [])
                            },
                            "needs_confirmation": True,
                            "is_draft": True,
                            "original_request": modified_draft
                        }
                    )
                else:
                    return AgentResponse(
                        success=False,
                        message="I couldn't modify the event. Please try again with different modifications.",
                        data=None
                    )
            
            # For new events, parse and validate times
            if start_time:
                event_start, event_end = await self._parse_time(start_time, duration_minutes)
                if end_time:
                    _, event_end = await self._parse_time(end_time)
            else:
                return AgentResponse(
                    success=False,
                    message="Please specify when you'd like to schedule the event (e.g., 'tomorrow at 2pm')",
                    data=None
                )
            
            if not event_start or not event_end:
                return AgentResponse(
                    success=False,
                    message="I couldn't understand the time you specified. Could you please provide the time in a format like 'Thursday at 4pm' or 'tomorrow at 2pm'?",
                    data={
                        "needs_time_clarification": True,
                        "original_request": {
                            "summary": summary,
                            "description": description,
                            "attendees": attendees,
                            "duration_minutes": duration_minutes
                        }
                    }
                )
            
            # Prepare the draft event
            draft_event = {
                "summary": summary,
                "description": description,
                "start_time": event_start.isoformat(),
                "end_time": event_end.isoformat(),
                "duration_minutes": duration_minutes,
                "attendees": attendees or []
            }
            
            # Store the current draft and show it
            self._current_draft = draft_event
            
            return AgentResponse(
                success=True,
                message="Here's the event I'm about to create. Please confirm if this is correct, or let me know what needs to be changed:",
                data={
                    "draft_event": {
                        "summary": summary,
                        "description": description,
                        "start": event_start.strftime("%Y-%m-%d %I:%M %p %Z"),
                        "end": event_end.strftime("%Y-%m-%d %I:%M %p %Z"),
                        "duration": f"{duration_minutes} minutes",
                        "attendees": attendees or []
                    },
                    "needs_confirmation": True,
                    "is_draft": True,
                    "original_request": draft_event
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating event: {str(e)}")
            if "time range is empty" in str(e):
                return AgentResponse(
                    success=False,
                    message="I couldn't create the event with the specified time. Could you please provide a different time?",
                    data={
                        "needs_time_clarification": True,
                        "original_request": {
                            "summary": summary,
                            "description": description,
                            "attendees": attendees,
                            "duration_minutes": duration_minutes
                        }
                    }
                )
            return AgentResponse(
                success=False,
                message=f"Error creating event: {str(e)}",
                data=None
            )
    
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
                    summary=params.get("summary", "New Event"),
                    start_time=params.get("start_time"),
                    end_time=params.get("end_time"),
                    description=params.get("description", ""),
                    attendees=params.get("attendees", []),
                    duration_minutes=params.get("duration", 30),
                    confirmed=params.get("confirmed", False),
                    is_modification=params.get("is_modification", False),
                    modifications=params.get("modifications", {})
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
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate the input data"""
        return (
            isinstance(input_data, dict) and
            "action" in input_data and
            isinstance(input_data.get("parameters", {}), dict)
        ) 