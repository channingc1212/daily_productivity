from typing import Dict, Any, List
from .base import BaseAgent, AgentResponse
from loguru import logger

# Define the manager agent that orchestrates all other agents
class ManagerAgent(BaseAgent):
    """Manager agent that orchestrates all other agents"""
    
    def __init__(self, config: Dict[str, Any]):
        # Set default model to gpt-4-mini if not specified
        config['model'] = config.get('model', 'gpt-4-mini')
        super().__init__(config)
        self.agents: Dict[str, BaseAgent] = {}
        
    def register_agent(self, name: str, agent: BaseAgent):
        """Register a new agent with the manager"""
        self.agents[name] = agent
        logger.info(f"Registered agent: {name}")
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agents"""
        return list(self.agents.keys())
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """Process input by delegating to appropriate agent"""
        if not self.validate_input(input_data):
            return AgentResponse(
                success=False,
                message="Invalid input data format",
                data=None
            )
        
        agent_name = input_data.get("agent")
        if agent_name not in self.agents:
            return AgentResponse(
                success=False,
                message=f"Agent {agent_name} not found",
                data={"available_agents": self.get_available_agents()}
            )
        
        try:
            result = await self.agents[agent_name].process(input_data)
            return result
        except Exception as e:
            logger.error(f"Error processing request with agent {agent_name}: {str(e)}")
            return AgentResponse(
                success=False,
                message=f"Error processing request: {str(e)}",
                data=None
            )
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate the input data"""
        return isinstance(input_data, dict) and "agent" in input_data 