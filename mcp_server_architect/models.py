#!/usr/bin/env python3
"""
Model definitions for Architect MCP Server.
"""

import logging
import os

# Configure logging
logger = logging.getLogger(__name__)

# Model dictionary mapping friendly names to PydanticAI model strings
MODEL_CONFIGS = {
    # Gemini models
    "gemini-2.5": "google-gla:gemini-2.5-pro-exp-03-25",
    
    # OpenAI models
    "gpt4o": "openai:gpt-4o",
}

# Default model to use for agent loops
DEFAULT_AGENT_MODEL = os.getenv("DEFAULT_AGENT_MODEL", "gpt4o")

# Default model to use for specific tasks - maintains Gemini for certain tools
DEFAULT_TASK_MODELS = {
    "generate_prd": os.getenv("PRD_MODEL", "gemini-2.5"),
    "think": os.getenv("THINK_MODEL", "gemini-2.5"),
}

def get_model_string(task: str = None) -> str:
    """
    Get the appropriate model string for a given task.
    
    Args:
        task: Optional task name to get a specific model for
        
    Returns:
        The PydanticAI model string to use
    """
    model_id = DEFAULT_TASK_MODELS.get(task, DEFAULT_AGENT_MODEL)
        
    if model_id not in MODEL_CONFIGS:
        logger.warning(f"Unknown model ID: {model_id}, falling back to {DEFAULT_AGENT_MODEL}")
        model_id = DEFAULT_AGENT_MODEL
        
    return MODEL_CONFIGS[model_id]