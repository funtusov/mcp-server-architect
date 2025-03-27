#!/usr/bin/env python3
"""
Core classes for the Architect AI MCP Server.
"""

import logging
import os

import google.generativeai as genai

from mcp_server_architect.file_context import FileContextBuilder

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configure Google Generative AI with API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.warning("GEMINI_API_KEY environment variable is not set. Gemini API calls will fail.")

# Get the default model from environment
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro-exp-03-25")


class ArchitectAI:
    """
    MCP server that acts as an AI Software Architect.
    Generates Product Requirements Documents (PRDs) based on codebase analysis.
    """

    # The tool will be registered with the FastMCP server in __main__.py
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

            # Call Gemini API
            client = genai.Client(api_key=api_key)
            model = client.get_model(DEFAULT_MODEL)
            response = model.generate_content(prompt)

            # Process and return the response
            return self._process_response(response)

        except Exception as e:
            logger.error(f"Error generating PRD: {str(e)}", exc_info=True)
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
