#!/usr/bin/env python3
"""
LLM tool for the Architect agent.
"""

import logging

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from mcp_server_architect.models import MODEL_CONFIGS
from mcp_server_architect.types import ArchitectDependencies

# Configure logging
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5"


class LLMInput(BaseModel):
    """Input schema for the LLM tool."""

    prompt: str = Field(..., description="The prompt to send to the LLM")
    model: str | None = Field(None, description="Optional model identifier (gemini-2.5 or gpt4o)")
    temperature: float | None = Field(0.7, description="Temperature for LLM generation (0.0-1.0)")
    max_tokens: int | None = Field(None, description="Maximum tokens to generate")


async def llm(ctx: RunContext[ArchitectDependencies], input_data: LLMInput) -> str:
    """
    Execute a prompt against a configured LLM.

    Args:
        ctx: The runtime context containing dependencies
        input_data: The input parameters specifying the prompt and generation parameters

    Returns:
        The text response from the LLM or an error message
    """
    try:
        # Determine which model to use - prefer the direct tool input if provided
        model_id = input_data.model if input_data.model else DEFAULT_MODEL

        # Use PydanticAI to handle the model selection
        logger.info(f"Using model ID: {model_id}")

        # Get the model string from our config or fallback to gemini-2.5
        model_string = MODEL_CONFIGS.get(model_id, MODEL_CONFIGS[DEFAULT_MODEL])

        # Create a simple agent for this single prompt
        agent = Agent(model_string, system_prompt="You are a helpful assistant.")

        response = await agent.run(
            input_data.prompt,
            generation_options={"temperature": input_data.temperature, "max_tokens": input_data.max_tokens},
        )

        return response.data

    except Exception as e:
        logger.error(f"Unexpected error in llm tool: {str(e)}", exc_info=True)
        return f"Error in LLM tool: {str(e)}"
