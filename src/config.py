import os
from typing import Dict, Any
from dotenv import load_dotenv
from loguru import logger

class Config:
    """Configuration management for the application"""
    
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        self._validate_required_env()
    
    def _validate_required_env(self):
        """Validate required environment variables"""
        required_vars = [
            "OPENAI_API_KEY",
            "GOOGLE_CLIENT_ID",
            "GOOGLE_CLIENT_SECRET",
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            logger.error(f"Missing required environment variables: {', '.join(missing)}")
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for specific agent"""
        base_config = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "model_name": os.getenv("OPENAI_MODEL_NAME", "o1-mini"),
        }
        
        # Add agent-specific configurations
        if agent_name == "email":
            base_config.update({
                "google_client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "google_client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
            })
        elif agent_name == "calendar":
            base_config.update({
                "google_client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "google_client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
            })
        
        return base_config 