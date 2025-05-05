"""
Configuration module for handling environment variables.

This module loads environment variables from the .env file and
provides structured access to OpenAI API configuration.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Find .env file and load environment variables
env_path = Path(__file__).parents[1] / ".env"
load_dotenv(dotenv_path=env_path)
logger.info(f"Environment variables loaded from {env_path}")


class OpenAIConfig:
    """Configuration settings for OpenAI API."""
    
    API_KEY: str = os.getenv("OPENAI_API_KEY", "your_api_key_here")
    MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "500"))
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate OpenAI configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise.
        
        Raises:
            ValueError: If API key is missing.
        """
        if not cls.API_KEY:
            logger.error("OPENAI_API_KEY environment variable is not set.")
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please add it to your .env file."
            )
        
        if cls.API_KEY == "your_api_key_here":
            logger.warning(
                "OPENAI_API_KEY is set to the placeholder value. "
                "Please replace it with your actual API key in the .env file."
            )
            return False
        
        logger.info("OpenAI configuration is valid.")
        return True


# Validate configuration on module import
try:
    valid_config = OpenAIConfig.validate()
    if not valid_config:
        logger.warning("OpenAI configuration is using placeholder values.")
except Exception as e:
    logger.error(f"Configuration validation failed: {e}")
