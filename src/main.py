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

@app.command()
def chat(ctx: typer.Context):
    """Start an interactive chat session with the assistant"""
    console.print(Panel.fit("Personal Assistant - Type 'exit' to quit"))
    
    async def chat_loop():
        manager = await setup_agents()
        
        while True:
            user_input = typer.prompt("You")
            
            if user_input.lower() == "exit":
                break
            
            try:
                # First, detect intent
                intent_response = await manager.process({
                    "agent": "intent_detector",
                    "user_input": user_input
                })
                
                if not intent_response.success:
                    console.print(f"[red]Error:[/red] {intent_response.message}")
                    continue
                
                # Parse the intent response
                try:
                    intent_data = json.loads(intent_response.data["raw_response"])
                    agent_name = intent_data["agent"]
                    
                    if agent_name not in ["email", "calendar"]:
                        console.print(f"[red]Error:[/red] Unknown agent: {agent_name}")
                        continue
                    
                    # Process with appropriate agent
                    agent_response = await manager.process({
                        "agent": agent_name,
                        "action": intent_data["action"],
                        "parameters": intent_data["parameters"]
                    })
                    
                    if agent_response.success:
                        console.print(f"[green]Success:[/green] {agent_response.message}")
                        if agent_response.data:
                            console.print(agent_response.data)
                    else:
                        console.print(f"[red]Error:[/red] {agent_response.message}")
                    
                except json.JSONDecodeError:
                    console.print("[red]Error:[/red] Failed to parse intent response")
                except KeyError as e:
                    console.print(f"[red]Error:[/red] Missing required field in intent: {str(e)}")
                
            except Exception as e:
                console.print(f"[red]Error:[/red] {str(e)}")
    
    try:
        asyncio.run(chat_loop())
    except KeyboardInterrupt:
        console.print("\nGoodbye! 👋")

if __name__ == "__main__":
    app() 