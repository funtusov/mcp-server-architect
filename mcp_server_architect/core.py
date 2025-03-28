#!/usr/bin/env python3
"""
Core classes for the Architect MCP Server.
"""

import logging

from mcp_server_architect.agents.manager import AgentManager

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Architect:
    """
    MCP server that acts as an AI Software Architect.
    Generates Product Requirements Documents (PRDs) based on codebase analysis
    and provides reasoning assistance for coding tasks.
    
    Uses an agent-based approach with PydanticAI to execute multiple actions 
    (web search, code reading, LLM calls) as needed to produce results.
    """
    
    def __init__(self):
        """Initialize the Architect with an agent manager."""
        self.agent_manager = AgentManager()
        logger.info("Initialized Architect with AgentManager")

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
        logger.info(f"Using codebase path: {codebase_path}")

        try:
            # Use the agent manager to run the agent loop
            return self.agent_manager.run_prd_agent(task_description, codebase_path)

        except Exception as e:
            # Log the exception with standard traceback
            logger.error(f"Unexpected error during PRD generation: {str(e)}", exc_info=True)
            return f"Error generating PRD: {str(e)}"


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
            # Use the agent manager to run the thinking agent
            return self.agent_manager.run_think_agent(request)

        except Exception as e:
            # Log the exception with standard traceback
            logger.error(f"Unexpected error during reasoning assistance: {str(e)}", exc_info=True)
            return f"Error providing reasoning assistance: {str(e)}"

