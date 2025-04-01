#!/usr/bin/env python3
"""
LLM tool for the Architect agent using litellm for multi-provider support.
"""

import logging

from litellm import acompletion
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)
from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from mcp_server_architect.types import ArchitectDependencies

# Configure logging
logger = logging.getLogger(__name__)

# Default model to use if none is specified
DEFAULT_MODEL = "gemini-2.5-pro"

# Mapping of user-friendly model names to litellm model strings
MODEL_MAPPING: dict[str, str] = {
    # OpenAI model
    "gpt4o": "openai/chatgpt-4o-latest",  # Don't change this to gpt4o
    # Google Gemini model
    "gemini-2.5-pro": "gemini/gemini-2.5-pro-exp-03-25",
    # Anthropic Claude model
    "claude-3.7-sonnet": "anthropic/claude-3-7-sonnet-20250219",
    # DeepSeek model (via OpenRouter)
    "deepseek-v3": "openrouter/deepseek/deepseek-chat-v3-0324",
}


class LLMInput(BaseModel):
    """Input schema for the LLM tool."""

    prompt: str = Field(..., description="The prompt to send to the LLM")
    model: str | None = Field(
        None,
        description="""
        Optional model identifier. Specify one of these model keys:
        
        - 'gpt4o' (DEFAULT): OpenAI's GPT-4o - Excellent multimodal model with strong reasoning, coding abilities, and creative tasks. Features fast processing speed and real-time thinking. Best for complex problems requiring deep understanding, coding, and general-purpose tasks.
        
        - 'gemini-2.5-pro': Google's Gemini 2.5 Pro - State-of-the-art reasoning model with structured thinking capabilities and advanced code generation. Excels at mathematics, science, and complex reasoning tasks. Top performer on benchmark leaderboards. Best for technical documentation, coding, and step-by-step problem-solving.
        
        - 'claude-3.7-sonnet': Anthropic's Claude 3.7 Sonnet - Features hybrid reasoning with transparent "thinking" processes. Superior at detailed analysis, long context understanding (up to 128K tokens), and careful instruction following. Best for nuanced tasks requiring critical thinking, comprehensive explanations, and content generation.
        
        - 'deepseek-v3': DeepSeek-v3-0324 - Specialized large-scale MoE (Mixture-of-Experts) model optimized for code generation and technical problem-solving. Exceptional at programming tasks, algorithm implementation, and debugging. Features fast response times and efficiency. Best for software engineering tasks, coding challenges, and technical implementations.
        """,
    )
    temperature: float | None = Field(
        None,
        description="Temperature for LLM generation (0.0-1.0). Lower values (0.0-0.3) for factual/deterministic responses, higher values (0.7-1.0) for creative/varied responses. Some models like Claude may override this setting when using thinking mode.",
    )


async def call_llm_core(prompt: str, model_id: str = None, temperature: float = None) -> str:
    """
    Core implementation for executing a prompt against an LLM using litellm.
    This function is the internal engine used by both the PydanticAI tool and the direct MCP tool.

    Args:
        prompt: The text prompt to send to the LLM
        model_id: The model identifier to use (e.g., gpt4o, gemini-2.5-pro)
        temperature: The temperature setting for generation

    Returns:
        The text response from the LLM or an error message
    """
    try:
        # Determine which model to use - prefer the direct tool input if provided
        model_id = model_id if model_id else DEFAULT_MODEL
        logger.info(f"Using model ID: {model_id}")

        # Get the litellm model string from our mapping or use default
        if model_id not in MODEL_MAPPING:
            logger.warning(f"Unknown model ID: {model_id}, falling back to {DEFAULT_MODEL}")
            model_id = DEFAULT_MODEL

        litellm_model = MODEL_MAPPING[model_id]
        logger.info(f"Using litellm model: {litellm_model}")

        # Prepare the messages
        messages = [{"role": "user", "content": prompt}]

        # Prepare the parameters
        params = {
            "model": litellm_model,
            "messages": messages,
            "temperature": temperature,
        }

        # Use minimal thinking params for Anthropic models in tests
        if litellm_model.startswith("anthropic/"):
            params["thinking"] = {"type": "enabled", "budget_tokens": 4096}
            params["max_tokens"] = 8192
            # With thinking enabled, temperature must be 1
            params["temperature"] = None

        # Make the API call with proper error handling
        # Add drop_params=True to handle incompatible parameters
        response = await acompletion(**params, drop_params=True)

        # Check if response has the expected structure
        if hasattr(response, "choices") and len(response.choices) > 0:
            return response.choices[0].message.content
        return f"Error: Unexpected response format from {model_id}: {response}"

    except AuthenticationError as e:
        error_msg = f"API key error with {model_id}: {str(e)}"
        logger.error(error_msg)
        return f"Error in LLM tool (authentication): {str(e)}"

    except APIConnectionError as e:
        error_msg = f"Connection error with {model_id}: {str(e)}"
        logger.error(error_msg)
        return f"Error in LLM tool (connection): {str(e)}"

    except RateLimitError as e:
        error_msg = f"Rate limit exceeded for {model_id}: {str(e)}"
        logger.error(error_msg)
        return f"Error in LLM tool (rate limit): {str(e)}"

    except BadRequestError as e:
        error_msg = f"Bad request error with {model_id}: {str(e)}"
        logger.error(error_msg)
        return f"Error in LLM tool (bad request): {str(e)}"

    except Exception as e:
        error_msg = f"Unexpected error in llm tool with {model_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return f"Error in LLM tool: {str(e)}"


async def llm(ctx: RunContext[ArchitectDependencies], input_data: LLMInput) -> str:
    """
    Execute a prompt against a specialized LLM model to perform targeted reasoning, problem-solving, or creative tasks.

    Use this tool when you need to leverage specific model strengths for a particular sub-task. Each model has different
    capabilities that may be more suitable for certain types of queries. For example:

    - Use GPT-4o for complex reasoning, creative writing, or general-purpose tasks
    - Use Gemini for up-to-date technical information, explanations, and code understanding
    - Use Claude for nuanced analysis, careful instructions, or processing lengthy context
    - Use DeepSeek for specialized code generation and technical implementation

    When to use this tool:
    - When you need specialized expertise beyond your capabilities
    - For complex sub-problems that benefit from targeted reasoning
    - When you want to compare multiple approaches to a problem
    - For generating code with specific requirements

    Args:
        ctx: The runtime context containing dependencies
        input_data: The input parameters including the prompt, model selection, and generation parameters

    Returns:
        The text response from the selected LLM or an error message if the request failed
    """
    return await call_llm_core(prompt=input_data.prompt, model_id=input_data.model, temperature=input_data.temperature)
