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

EVENT_ANALYSIS_PROMPT = """You are a calendar assistant analyzing event details and providing suggestions.
Current event details:
{event_details}

Previous events in context:
{context_events}

Analyze the event and provide:
1. Potential conflicts or overlaps
2. Suggested modifications (duration, time, etc.)
3. Related events or patterns
4. Smart defaults (based on user patterns)
5. Scheduling optimization suggestions

Return ONLY a JSON object in the following format:
{{
    "analysis": {{
        "conflicts": [
            {{
                "type": "overlap",
                "with_event": "event name",
                "time": "event time"
            }}
        ],
        "suggestions": [
            {{
                "type": "duration",
                "suggestion": "Consider extending to 1 hour",
                "reason": "Most similar events are 1 hour long"
            }}
        ],
        "related_events": [
            {{
                "name": "event name",
                "relationship": "similar time/day"
            }}
        ],
        "patterns": [
            {{
                "type": "timing",
                "pattern": "Usually schedule shopping in the afternoon"
            }}
        ],
        "optimizations": [
            {{
                "type": "schedule",
                "suggestion": "Better time slot available at 2 PM"
            }}
        ]
    }}
}}"""

MODIFICATION_PROMPT = """You are a calendar assistant handling event modifications.
Current event draft:
{current_draft}

User's modification request:
{modification_request}

Previous modifications:
{modification_history}

Analyze the request and determine:
1. What specific aspects need to be modified
2. Whether the modification is relative or absolute
3. Any implied changes that should be made
4. Any clarifications needed from the user

For duration changes:
- "extend to X hour/minutes" -> absolute duration (e.g., "duration_minutes": 60)
- "extend by X hour/minutes" -> relative duration (e.g., "duration_minutes": "+=30")
- "make it X hour/minutes longer" -> relative duration (e.g., "duration_minutes": "+=60")
- "change to X hour/minutes" -> absolute duration (e.g., "duration_minutes": 45)
- "set duration to X" -> absolute duration (e.g., "duration_minutes": 90)

Return ONLY a JSON object in the following format:
{{
    "modifications": {{
        "explicit": {{
            "duration_minutes": 60
        }},
        "implicit": {{}},
        "requires_clarification": false,
        "clarification_question": null
    }}
}}"""

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
        self.event_analysis_prompt = ChatPromptTemplate.from_template(EVENT_ANALYSIS_PROMPT)
        self.modification_prompt = ChatPromptTemplate.from_template(MODIFICATION_PROMPT)
        self._current_draft = None  # Store the current draft event
        self._draft_id = 0  # Initialize draft ID counter
        self._modification_history = []
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

    async def _analyze_event(self, event_details: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to analyze event details and provide suggestions"""
        try:
            # Get context events (events around the same time)
            context_events = await self._get_context_events(event_details)
            
            # Analyze using LLM
            chain = self.event_analysis_prompt | self.llm
            result = await chain.ainvoke({
                "event_details": json.dumps(event_details, indent=2),
                "context_events": json.dumps(context_events, indent=2)
            })
            
            # Clean and parse the response
            response_text = result.content.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            try:
                return json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing analysis response: {str(e)}")
                # Return empty analysis structure if parsing fails
                return {
                    "analysis": {
                        "conflicts": [],
                        "suggestions": [],
                        "related_events": [],
                        "patterns": [],
                        "optimizations": []
                    }
                }
                
        except Exception as e:
            logger.error(f"Error analyzing event: {str(e)}")
            return {
                "analysis": {
                    "conflicts": [],
                    "suggestions": [],
                    "related_events": [],
                    "patterns": [],
                    "optimizations": []
                }
            }

    async def _get_context_events(self, event_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get events around the same time for context"""
        try:
            start_time = datetime.fromisoformat(event_details["start_time"])
            end_time = datetime.fromisoformat(event_details["end_time"])
            
            # Look at events 1 day before and after
            time_min = (start_time - timedelta(days=1)).isoformat()
            time_max = (end_time + timedelta(days=1)).isoformat()
            
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=time_min,
                timeMax=time_max,
                maxResults=10,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            return events_result.get('items', [])
        except Exception as e:
            logger.error(f"Error getting context events: {str(e)}")
            return []

    async def _handle_modification(self, current_draft: Dict[str, Any], modification_request: str) -> Dict[str, Any]:
        """Use LLM to handle event modifications"""
        try:
            # Get modification analysis from LLM
            chain = self.modification_prompt | self.llm
            result = await chain.ainvoke({
                "current_draft": json.dumps(current_draft, indent=2),
                "modification_request": modification_request,
                "modification_history": json.dumps(self._modification_history, indent=2)
            })
            
            # Clean and parse the response
            response_text = result.content.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            try:
                modifications = json.loads(response_text)
                
                # Validate the response structure
                if not isinstance(modifications, dict) or "modifications" not in modifications:
                    logger.error("Invalid modification response structure")
                    return None
                
                # Apply explicit modifications
                modified_draft = current_draft.copy()
                explicit_mods = modifications.get("modifications", {}).get("explicit", {})
                implicit_mods = modifications.get("modifications", {}).get("implicit", {})
                
                # Handle duration modifications
                if "duration_minutes" in explicit_mods:
                    new_duration = explicit_mods["duration_minutes"]
                    if isinstance(new_duration, str) and new_duration.startswith("+="):
                        try:
                            # Handle relative duration changes
                            additional_minutes = int(new_duration[2:])
                            current_duration = modified_draft.get("duration_minutes", 30)
                            new_duration = current_duration + additional_minutes
                        except ValueError:
                            logger.error("Invalid relative duration format")
                            return None
                    
                    try:
                        # Ensure duration is an integer
                        new_duration = int(float(new_duration))
                        modified_draft["duration_minutes"] = new_duration
                        start_time = datetime.fromisoformat(modified_draft["start_time"])
                        modified_draft["end_time"] = (start_time + timedelta(minutes=new_duration)).isoformat()
                    except (ValueError, TypeError):
                        logger.error("Invalid duration value")
                        return None
                
                # Handle other explicit modifications
                for key in ["summary", "description", "attendees"]:
                    if key in explicit_mods:
                        modified_draft[key] = explicit_mods[key]
                
                # Handle implicit modifications
                for key, value in implicit_mods.items():
                    modified_draft[key] = value
                
                # Store modification history
                self._modification_history.append({
                    "request": modification_request,
                    "changes": {
                        "explicit": explicit_mods,
                        "implicit": implicit_mods
                    }
                })
                
                return modified_draft
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing modification response: {str(e)}")
                logger.error(f"Raw response: {response_text}")
                return None
            
        except Exception as e:
            logger.error(f"Error handling modification: {str(e)}")
            return None

    async def _create_event(self, parameters: Dict[str, Any]) -> AgentResponse:
        """Create a new calendar event draft"""
        try:
            # Get current date for reference
            now = datetime.now(self.timezone)
            tomorrow = now + timedelta(days=1)
            
            # Parse the start time
            start_time_str = parameters.get("start_time", "")
            try:
                start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M")
                # Add timezone info
                start_time = start_time.replace(tzinfo=self.timezone)
                
                # For "tomorrow" references, ensure it's actually tomorrow
                if "tomorrow" in parameters.get("description", "").lower():
                    start_time = start_time.replace(
                        year=tomorrow.year,
                        month=tomorrow.month,
                        day=tomorrow.day
                    )
                # For any other case, ensure the date is in the near future
                else:
                    # If date is in the past or too far in the future, adjust it
                    if abs((start_time.date() - now.date()).days) > 7:
                        # If more than a week away, assume it should be this week or next
                        target_day = start_time.weekday()
                        current_day = now.weekday()
                        days_ahead = target_day - current_day
                        if days_ahead <= 0:  # If target day has passed this week
                            days_ahead += 7  # Move to next week
                        target_date = now.date() + timedelta(days=days_ahead)
                        start_time = start_time.replace(
                            year=target_date.year,
                            month=target_date.month,
                            day=target_date.day
                        )
                    elif start_time < now:
                        # If in the past but within a week, move to tomorrow
                        start_time = start_time.replace(
                            year=tomorrow.year,
                            month=tomorrow.month,
                            day=tomorrow.day
                        )
                
            except ValueError:
                return AgentResponse(
                    success=False,
                    message="Invalid start time format. Please specify when you'd like to schedule the event.",
                    data=None
                )

            # Calculate end time
            duration_mins = parameters.get("duration_minutes", 30)
            end_time = start_time + timedelta(minutes=duration_mins)

            # Format times consistently
            time_format = "%Y-%m-%d %I:%M %p"  # Remove timezone from format string
            
            # Create the draft event
            self._draft_id += 1
            self._current_draft = {
                "summary": parameters.get("summary", "New Event"),
                "description": parameters.get("description", ""),
                "start": start_time.strftime(time_format),
                "end": end_time.strftime(time_format),
                "duration": f"{duration_mins} minutes",
                "attendees": parameters.get("attendees", [])
            }

            return AgentResponse(
                success=True,
                message="Here's the event I'm about to create. Please confirm if this is correct, or let me know what needs to be changed:",
                data={
                    "draft_event": self._current_draft,
                    "analysis": {
                        "conflicts": [],
                        "suggestions": [
                            {
                                "type": "duration",
                                "suggestion": "Consider extending to 1 hour",
                                "reason": "Most similar events are 1 hour long"
                            }
                        ],
                        "related_events": [],
                        "patterns": [
                            {
                                "type": "timing",
                                "pattern": "Usually schedule shopping in the afternoon"
                            }
                        ],
                        "optimizations": [
                            {
                                "type": "schedule",
                                "suggestion": "Better time slot available at 2 PM"
                            }
                        ]
                    },
                    "needs_confirmation": True,
                    "is_draft": True
                }
            )
        except Exception as e:
            logger.error(f"Error creating event: {str(e)}")
            return AgentResponse(
                success=False,
                message=f"Failed to create event: {str(e)}",
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
                return await self._create_event(params)
            elif action == "modify_draft":
                return await self._modify_draft(params)
            elif action == "confirm_draft":
                return await self._confirm_draft()
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

    async def _modify_draft(self, parameters: Dict[str, Any]) -> AgentResponse:
        """Modify the current draft event"""
        if not self._current_draft:
            return AgentResponse(
                success=False,
                message="No draft event to modify. Please create an event first.",
                data=None
            )

        try:
            modification = parameters.get("modification", {})
            mod_type = modification.get("type")
            mod_value = modification.get("value")

            if mod_type == "duration":
                # Convert duration to minutes if needed
                duration_mins = int(mod_value)
                # Use consistent time format
                time_format = "%Y-%m-%d %I:%M %p"  # Remove timezone from format string
                start_time = datetime.strptime(self._current_draft["start"], time_format)
                # Add timezone info
                start_time = start_time.replace(tzinfo=self.timezone)
                end_time = start_time + timedelta(minutes=duration_mins)
                
                self._current_draft.update({
                    "end": end_time.strftime(time_format),
                    "duration": f"{duration_mins} minutes"
                })
            
            return AgentResponse(
                success=True,
                message="Here's the updated event. Please confirm if this is correct:",
                data={
                    "draft_event": self._current_draft,
                    "needs_confirmation": True,
                    "is_draft": True
                }
            )
        except Exception as e:
            logger.error(f"Error modifying draft: {str(e)}")
            return AgentResponse(
                success=False,
                message=f"Failed to modify draft: {str(e)}",
                data=None
            )

    async def _confirm_draft(self) -> AgentResponse:
        """Confirm and create the current draft event"""
        if not self._current_draft:
            return AgentResponse(
                success=False,
                message="No draft event to confirm. Please create an event first.",
                data=None
            )

        try:
            # Here you would actually create the event in Google Calendar
            # For now, we'll just simulate success
            confirmed_event = self._current_draft.copy()
            self._current_draft = None  # Clear the draft
            
            return AgentResponse(
                success=True,
                message="Event created successfully!",
                data={"event": confirmed_event}
            )
        except Exception as e:
            logger.error(f"Error confirming event: {str(e)}")
            return AgentResponse(
                success=False,
                message=f"Failed to create event: {str(e)}",
                data=None
            ) 