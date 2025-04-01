#!/usr/bin/env python3
"""
Entry point for the mcp-server-architect package.
"""

import argparse
import logging
import os

import logfire
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic_ai import Agent

from mcp_server_architect.core import Architect
from mcp_server_architect.tools.llm import call_llm_core
from mcp_server_architect.version import __version__

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Software Architect MCP Server that generates PRDs based on codebase analysis",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"mcp-server-architect {__version__}",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--dotenv",
        type=str,
        help="Path to .env file",
        default=".env",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        help="Gemini model to use",
        default=os.getenv("GEMINI_MODEL", "gemini-2.5-pro-exp-03-25"),
    )
    return parser.parse_args()


def main():
    """Main entry point for the MCP server."""
    args = parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Load environment variables from .env file if it exists
    dotenv_path = args.dotenv
    if os.path.exists(dotenv_path):
        logger.info(f"Loading environment variables from {dotenv_path}")
        load_dotenv(dotenv_path)
    else:
        logger.warning(f".env file not found at {dotenv_path}. Using environment variables.")

    # Configure Logfire for instrumentation if API key is available
    logfire_api_key = os.getenv("LOGFIRE_API_KEY")
    if logfire_api_key:
        logger.info("Configuring Logfire instrumentation")
        logfire.configure(token=logfire_api_key)
        # Instrument PydanticAI Agent to capture all agent activity
        Agent.instrument_all()
    else:
        logger.warning("LOGFIRE_API_KEY not found. Logfire instrumentation disabled.")

    # Check for API keys required by litellm
    warnings = []

    # Check for OpenAI API key (required for GPT-4o)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        warnings.append("OPENAI_API_KEY")

    # Check for Google/Gemini API key
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        warnings.append("GEMINI_API_KEY")

    # Check for Anthropic API key
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        warnings.append("ANTHROPIC_API_KEY")

    # Check for OpenRouter API key
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        warnings.append("OPENROUTER_API_KEY")

    # Log warnings for missing API keys
    if warnings:
        logger.warning(f"The following API keys are not set: {', '.join(warnings)}. Some models will not be available.")

    # Set Gemini model from arguments or environment
    os.environ["GEMINI_MODEL"] = args.gemini_model
    logger.info(f"Using Gemini model: {os.environ['GEMINI_MODEL']}")

    # Create MCP server instance
    server = FastMCP(
        "Architect",
        description="AI Software Architect that generates PRDs and design documents based on codebase analysis, and provides reasoning assistance",
    )

    # Register the Architect tools using decorator pattern
    architect = Architect()

    @server.tool()
    async def generate_prd(task_description: str, codebase_path: str) -> str:
        """
        Generate a PRD or high-level design document based on codebase analysis and task description.

        Args:
            task_description (str): Detailed description of the programming task
            codebase_path (str): Path to the local codebase directory

        Returns:
            str: The generated PRD or design document
        """
        return await architect.generate_prd(task_description, codebase_path)

    @server.tool()
    async def think(request: str, codebase_path: str = "") -> str:
        """
        Provide deep reasoning assistance for a project-related question or issue.

        Args:
            request (str): Detailed description of the coding task/issue and relevant code snippets
            codebase_path (str): Path to the local codebase directory

        Returns:
            str: Reasoning guidance and potential solutions
        """
        return await architect.think(request, codebase_path)

    @server.tool()
    async def llm(prompt: str, model: str = None, temperature: float = None) -> str:
        """
        Execute a prompt against a specialized LLM model directly.

        Args:
            prompt: The text prompt to send to the LLM
            model: Optional model identifier (gpt4o, gemini-2.5-pro, claude-3.7-sonnet, deepseek-v3)
            temperature: Optional temperature setting (0.0-1.0)

        Returns:
            The text response from the selected LLM

        Available models:
        - gpt4o (DEFAULT): OpenAI's GPT-4o - Excellent for reasoning, coding, and creative tasks
        - gemini-2.5-pro: Google's Gemini 2.5 Pro - Great for technical content and structured reasoning
        - claude-3.7-sonnet: Anthropic's Claude - Superior for detailed analysis and careful reasoning
        - deepseek-v3: DeepSeek - Specialized for coding and technical problem-solving
        """
        logger.info(f"Direct LLM call via MCP: model={model}, prompt_length={len(prompt)}")
        return await call_llm_core(prompt, model, temperature)

    # Start the MCP server
    logger.info(f"Starting Architect MCP server v{__version__}...")
    server.run()


if __name__ == "__main__":
    main()
