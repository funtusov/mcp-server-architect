#!/usr/bin/env python3
"""
Core classes for the Architect MCP Server.
"""

import logging
import os
import time
from typing import Any

from google import genai
from google.genai import errors as genai_errors

from mcp_server_architect.file_context import FileContextBuilder

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configure Google Generative AI with API key
api_key = os.getenv("GEMINI_API_KEY")
logger.info(f"GEMINI_API_KEY environment check: {'Set' if api_key else 'Not set'}")
if not api_key:
    logger.warning("GEMINI_API_KEY environment variable is not set. Gemini API calls will fail.")

# Get the default model from environment
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro-exp-03-25")


class Architect:
    """
    MCP server that acts as an AI Software Architect.
    Generates Product Requirements Documents (PRDs) based on codebase analysis
    and provides reasoning assistance for coding tasks.
    """

    # The tools will be registered with the FastMCP server in __main__.py
    def generate_prd(self, task_description: str, codebase_path: str) -> str:
        """
        Generate a PRD or high-level design document based on codebase analysis and task description.

        Args:
            task_description (str): Detailed description of the programming task
            codebase_path (str): Path to the local codebase directory

        Returns:
            str: The generated PRD or design document
        """
        logger.info(f"Generating PRD for task: {task_description[:50]}...")
        logger.info(f"Analyzing codebase at: {codebase_path}")

        try:
            # Build context from codebase files
            context_builder = FileContextBuilder(codebase_path)
            code_context = context_builder.build_context()

            # Create the prompt for Gemini
            prompt = self._create_prompt(task_description, code_context)

            # Generate the response
            return self._generate_gemini_response(prompt, "PRD generation")

        except Exception as e:
            # Log the exception with standard traceback
            logger.error(f"Unexpected error during PRD generation: {str(e)}", exc_info=True)
            return f"Error generating PRD: {str(e)}"

    def _create_prompt(self, task_description: str, code_context: str) -> str:
        """
        Create a comprehensive prompt for the Gemini model.

        Args:
            task_description (str): The task description
            code_context (str): The extracted code context

        Returns:
            str: The formatted prompt
        """
        return f"""
        You are an expert software architect and technical lead.
        
        Your task is to create a Product Requirements Document (PRD) or High-Level Design Document based on the following:
        
        ## Task Description:
        {task_description}
        
        ## Existing Codebase Context:
        ```
        {code_context}
        ```
        
        Your PRD should include:
        1. Overview of the requested feature/task
        2. Technical requirements and constraints
        3. Proposed architecture/design
        4. Implementation plan with specific files to modify
        5. Potential challenges and mitigations
        
        Format your response in markdown. Be concise but comprehensive.
        """

    def _generate_gemini_response(self, prompt: str, operation_type: str) -> str:
        """
        Generate a response from Gemini API with error handling.

        Args:
            prompt (str): The prompt to send to Gemini
            operation_type (str): Type of operation (for error messages)

        Returns:
            str: The processed response text
        """
        try:
            # Call Gemini API with retry logic
            response = self._call_gemini_api(prompt)

            # Process and return the response
            return self._process_response(response)

        except genai_errors.ServerError as e:
            # Log essential error information
            error_str = str(e)
            logger.error(f"Server error from Gemini API during {operation_type}: {error_str}", exc_info=True)

            # Concise service error message
            return f"The API returned an error: {error_str}."

    def _call_gemini_api(self, prompt: str, max_retries: int = 3, retry_delay: int = 10) -> Any:
        """
        Call the Gemini API with automatic retries on server errors.

        Args:
            prompt (str): The prompt to send to Gemini
            max_retries (int): Maximum number of retry attempts
            retry_delay (int): Delay between retries in seconds

        Returns:
            Any: The Gemini API response

        Raises:
            genai_errors.ServerError: If all retry attempts fail
            Exception: For any other errors
        """
        client = genai.Client(api_key=api_key)

        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(f"Gemini API call attempt {attempt}/{max_retries}")
                return client.models.generate_content(model=DEFAULT_MODEL, contents=prompt)
            except genai_errors.ServerError as e:
                logger.warning(f"Server error on attempt {attempt}/{max_retries}: {str(e)}")

                # If this is the last attempt, re-raise the exception
                if attempt == max_retries:
                    logger.error("All retry attempts failed")
                    raise

                # Wait before the next retry
                logger.info(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)

        # This line should never be reached as the loop either returns or raises an exception
        return None

    def _process_response(self, response) -> str:
        """
        Process the response from the Gemini model.

        Args:
            response: The response from the Gemini model

        Returns:
            str: The processed response
        """
        # Extract the text from the response
        if hasattr(response, "text"):
            return response.text
        if hasattr(response, "parts"):
            return "".join(part.text for part in response.parts)
        return str(response)

    def think(self, request: str) -> str:
        """
        Provide reasoning assistance for a stuck LLM on a coding task.

        Args:
            request (str): Detailed description of the coding task/issue and relevant code

        Returns:
            str: Reasoning guidance and potential solutions
        """
        logger.info(f"Providing reasoning assistance for request: {request[:50]}...")

        try:
            # Create the prompt for Gemini
            prompt = self._create_think_prompt(request)

            # Generate the response
            return self._generate_gemini_response(prompt, "reasoning assistance")

        except Exception as e:
            # Log the exception with standard traceback
            logger.error(f"Unexpected error during reasoning assistance: {str(e)}", exc_info=True)
            return f"Error providing reasoning assistance: {str(e)}"

    def _create_think_prompt(self, request: str) -> str:
        """
        Create a comprehensive prompt for the Gemini model to provide reasoning assistance.

        Args:
            request (str): The detailed request including task description and code snippets

        Returns:
            str: The formatted prompt
        """
        return f"""
        You are an expert software developer with deep expertise in code analysis and problem-solving.
        
        You need to help another AI that is stuck on a coding task. Analyze the request below and provide
        your reasoning, insights, and potential solutions.
        
        ## Request:
        {request}
        
        In your response:
        1. Identify the core problem or challenge
        2. Break down the problem into manageable steps
        3. Provide specific coding approaches or patterns to resolve the issue
        4. Suggest alternative solutions when appropriate
        5. Explain your reasoning clearly
        
        Format your response in markdown. Be concise but thorough. Focus on practical, implementable solutions.
        """
