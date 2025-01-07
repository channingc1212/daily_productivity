import asyncio
import typer
import json
from rich.console import Console
from rich.panel import Panel
from loguru import logger

from config import Config
from agents.manager import ManagerAgent
from agents.intent_detector import IntentDetectorAgent
from agents.email_agent import EmailAgent
from agents.calendar_agent import CalendarAgent
from utils.logging import setup_logging

app = typer.Typer()
console = Console()

async def setup_agents():
    """Initialize and setup all agents"""
    try:
        config = Config()
        
        # Initialize manager
        manager = ManagerAgent(config.get_agent_config("manager"))
        
        # Initialize and register intent detector
        intent_detector = IntentDetectorAgent(config.get_agent_config("intent_detector"))
        manager.register_agent("intent_detector", intent_detector)
        
        # Initialize and register email agent
        email_agent = EmailAgent(config.get_agent_config("email"))
        manager.register_agent("email", email_agent)
        
        # Initialize and register calendar agent
        calendar_agent = CalendarAgent(config.get_agent_config("calendar"))
        manager.register_agent("calendar", calendar_agent)
        
        return manager
    except Exception as e:
        logger.error(f"Error setting up agents: {str(e)}")
        raise

def main():
    """Main entry point for the application"""
    # Setup logging
    setup_logging()
    logger.info("Starting personal assistant...")
    
    welcome_message = """Welcome to Personal Assistant!

Commands:
- Type your request in natural language
- Type 'exit' or 'quit' to close the application

Examples:
- "Show me my recent emails"
- "Send an email to someone@example.com"
- "Show my calendar for next week"
- "Schedule a meeting tomorrow at 2pm"
"""
    
    console.print(Panel.fit(welcome_message))
    
    async def chat_loop():
        manager = await setup_agents()
        logger.info("All agents initialized successfully")
        
        while True:
            user_input = typer.prompt("You")
            
            if user_input.lower() in ["exit", "quit"]:
                logger.info("User requested to exit")
                console.print("[green]Goodbye! Have a great day! 👋[/green]")
                break
            
            try:
                logger.debug(f"Processing user input: {user_input}")
                
                # First, detect intent
                intent_response = await manager.process({
                    "agent": "intent_detector",
                    "user_input": user_input
                })
                
                if not intent_response.success:
                    logger.error(f"Intent detection failed: {intent_response.message}")
                    console.print(f"[red]Error:[/red] {intent_response.message}")
                    continue
                
                # Parse the intent response
                try:
                    raw_response = intent_response.data["raw_response"].strip()
                    logger.debug(f"Raw intent response: {raw_response}")
                    
                    # Remove any markdown code block formatting if present
                    if raw_response.startswith("```"):
                        raw_response = raw_response.split("```")[1]
                        if raw_response.startswith("json"):
                            raw_response = raw_response[4:]
                    raw_response = raw_response.strip()
                    
                    logger.debug(f"Cleaned intent response: {raw_response}")
                    
                    intent_data = json.loads(raw_response)
                    agent_name = intent_data["agent"]
                    
                    if agent_name not in ["email", "calendar"]:
                        logger.error(f"Unknown agent requested: {agent_name}")
                        console.print(f"[red]Error:[/red] Unknown agent: {agent_name}")
                        continue
                    
                    logger.info(f"Routing request to {agent_name} agent")
                    logger.debug(f"Intent data: {intent_data}")
                    
                    # Process with appropriate agent
                    agent_response = await manager.process({
                        "agent": agent_name,
                        "action": intent_data["action"],
                        "parameters": intent_data["parameters"]
                    })
                    
                    if agent_response.success:
                        logger.info(f"Request processed successfully by {agent_name} agent")
                        console.print(f"[green]Success:[/green] {agent_response.message}")
                        if agent_response.data:
                            console.print(agent_response.data)
                    else:
                        logger.error(f"Agent processing failed: {agent_response.message}")
                        console.print(f"[red]Error:[/red] {agent_response.message}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse intent response JSON: {str(e)}")
                    logger.error(f"Raw response was: {intent_response.data['raw_response']}")
                    console.print("[red]Error:[/red] Failed to understand the request. Please try rephrasing.")
                except KeyError as e:
                    logger.error(f"Missing key in intent response: {str(e)}")
                    console.print("[red]Error:[/red] Invalid intent response format. Please try again.")
                except Exception as e:
                    logger.error(f"Error processing intent: {str(e)}")
                    console.print("[red]Error:[/red] An unexpected error occurred. Please try again.")
                
            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}")
                console.print(f"[red]Error:[/red] An unexpected error occurred. Please try again.")
    
    try:
        asyncio.run(chat_loop())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        console.print("\n[yellow]Please use 'exit' or 'quit' to close the application next time.[/yellow]")
        console.print("[green]Goodbye! 👋[/green]")

if __name__ == "__main__":
    main() 