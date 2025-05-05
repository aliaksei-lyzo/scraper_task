"""
Tests for the config module.
"""
import os
import pytest
from unittest.mock import patch

from src.config import OpenAIConfig


def test_openai_config_default_values():
    """Test that OpenAIConfig loads default values correctly."""
    assert isinstance(OpenAIConfig.API_KEY, str) or OpenAIConfig.API_KEY is None
    assert isinstance(OpenAIConfig.MODEL, str)
    assert isinstance(OpenAIConfig.TEMPERATURE, float)
    assert isinstance(OpenAIConfig.MAX_TOKENS, int)


@patch.dict(os.environ, {
    "OPENAI_API_KEY": "test_key_123",
    "OPENAI_MODEL": "gpt-4",
    "OPENAI_TEMPERATURE": "0.5",
    "OPENAI_MAX_TOKENS": "2000"
}, clear=True)
def test_openai_config_custom_values():
    """Test that OpenAIConfig reads custom environment values."""
    # Re-import to reload environment variables
    from importlib import reload
    import src.config
    reload(src.config)
    
    assert src.config.OpenAIConfig.API_KEY == "test_key_123"
    assert src.config.OpenAIConfig.MODEL == "gpt-4"
    assert src.config.OpenAIConfig.TEMPERATURE == 0.5
    assert src.config.OpenAIConfig.MAX_TOKENS == 2000


def test_validate_without_api_key():
    """Test validation raises error with missing API key."""
    with patch.object(OpenAIConfig, "API_KEY", None):
        with pytest.raises(ValueError, match="OPENAI_API_KEY.*not set"):
            OpenAIConfig.validate()


def test_validate_with_placeholder_api_key():
    """Test validation returns False with placeholder API key."""
    with patch.object(OpenAIConfig, "API_KEY", "your_api_key_here"):
        assert OpenAIConfig.validate() is False
