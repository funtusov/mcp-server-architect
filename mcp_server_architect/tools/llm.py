#!/usr/bin/env python3
"""
LLM tool for the Architect agent.
"""

import logging
import os
import time

from google import genai
from google.genai import errors as genai_errors
from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from mcp_server_architect.types import ArchitectDependencies

# Configure logging
logger = logging.getLogger(__name__)

# Default model from environment
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro-exp-03-25")


class LLMInput(BaseModel):
    """Input schema for the LLM tool."""
    prompt: str = Field(..., description="The prompt to send to the LLM")
    model: str | None = Field(None, description="Optional override for the LLM model to use")
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
        # Use environment variables directly for API keys
        # Note: The genai.configure() will use the GEMINI_API_KEY environment variable 
        # if not explicitly provided
        
        # Determine which model to use
        model = input_data.model if input_data.model else DEFAULT_MODEL
        logger.info(f"Using LLM model: {model}")
        
        # Call Gemini with retry logic
        try:
            response = _call_gemini_api(
                prompt=input_data.prompt, 
                model=model,
                temperature=input_data.temperature,
                max_tokens=input_data.max_tokens
            )
            
            # Process and return the response
            return _process_response(response)
            
        except genai_errors.ServerError as e:
            error_str = str(e)
            logger.error(f"Server error from Gemini API: {error_str}", exc_info=True)
            return f"The API returned an error: {error_str}."
        
    except Exception as e:
        logger.error(f"Unexpected error in llm tool: {str(e)}", exc_info=True)
        return f"Error in LLM tool: {str(e)}"


def _call_gemini_api(prompt: str, model: str = DEFAULT_MODEL, 
                   temperature: float = 0.7, max_tokens: int | None = None, 
                   max_retries: int = 3, retry_delay: int = 10) -> any:
    """
    Call the Gemini API with automatic retries on server errors.
    
    Args:
        prompt: The prompt to send to Gemini
        model: The model to use
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        The Gemini API response
        
    Raises:
        genai_errors.ServerError: If all retry attempts fail
        Exception: For any other errors
    """
    # Configure the client - genai will use GEMINI_API_KEY from environment
    client = genai.GenerativeModel(
        model_name=model,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens
        }
    )
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"Gemini API call attempt {attempt}/{max_retries}")
            return client.generate_content(prompt)
        except genai_errors.ServerError as e:
            logger.warning(f"Server error on attempt {attempt}/{max_retries}: {str(e)}")
            
            # If this is the last attempt, re-raise the exception
            if attempt == max_retries:
                logger.error("All retry attempts failed")
                raise
            
            # Wait before the next retry
            logger.info(f"Waiting {retry_delay} seconds before retry...")
            time.sleep(retry_delay)
    
    # This line should never be reached
    return None


def _process_response(response) -> str:
    """
    Process the response from the Gemini model.
    
    Args:
        response: The response from the Gemini model
        
    Returns:
        str: The processed text response
    """
    # Extract the text from the response
    if hasattr(response, "text"):
        return response.text
    if hasattr(response, "parts"):
        return "".join(part.text for part in response.parts)
    return str(response)