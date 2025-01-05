from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel

# Define the response format for all agents
class AgentResponse(BaseModel):
    """Standard response format for all agents"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

# Define the base class for all agents
class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """Process the input and return a response"""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate the input data"""
        pass 