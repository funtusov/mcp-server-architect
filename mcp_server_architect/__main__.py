#!/usr/bin/env python3
"""
Entry point for the mcp-server-architect package.
"""

import os
import sys
import logging
import argparse
from dotenv import load_dotenv

from mcp import MCP, Config, run
from mcp_server_architect.core import ArchitectAI
from mcp_server_architect.version import __version__

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Software Architect MCP Server that generates PRDs based on codebase analysis"
    )
    parser.add_argument(
        "--version", action="version", version=f"mcp-server-architect {__version__}"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--dotenv", type=str, help="Path to .env file", default=".env"
    )
    parser.add_argument(
        "--gemini-model", type=str, help="Gemini model to use", 
        default=os.getenv("GEMINI_MODEL", "gemini-2.5-pro-exp-03-25")
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
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is required")
        sys.exit(1)
        
    # Set GOOGLE_API_KEY for Gemini SDK (required by the SDK)
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Set Gemini model from arguments or environment
    os.environ["GEMINI_MODEL"] = args.gemini_model
    logger.info(f"Using Gemini model: {os.environ['GEMINI_MODEL']}")
    
    # Create MCP server configuration
    config = Config(
        name="ArchitectAI",
        description="AI Software Architect that generates PRDs and design documents based on codebase analysis",
    )
    
    # Create MCP server instance
    server = MCP(config)
    
    # Register the ArchitectAI class and its methods
    server.register(ArchitectAI())
    
    # Start the MCP server
    logger.info("Starting ArchitectAI MCP server...")
    run(server)


if __name__ == "__main__":
    main()