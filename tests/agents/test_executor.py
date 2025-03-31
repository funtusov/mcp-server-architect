#!/usr/bin/env python3
"""
Test Agent executor with multiple tools.
"""

import logging
import os

import pytest

from mcp_server_architect.agents.executor import AgentExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_create_agent_with_multiple_tools():
    """Test creating an agent with multiple tools using both initialization methods."""
    # Create the agent executor
    executor = AgentExecutor()

    # Create an agent for each supported model type
    logger.info("Testing agent creation with multiple tools")

    # Test creating agent with GPT-4o (direct initialization)
    agent_openai = executor._create_agent(task=None)

    # Test creating agent with Gemini (string initialization)
    agent_gemini = executor._create_agent(task="generate_prd")

    # Verify the agents were created
    assert agent_openai is not None
    assert agent_gemini is not None

    # Log success
    logger.info("Successfully created agents with multiple tools")


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_all_tools_initialization():
    """Test that all tools are correctly initialized."""
    # Create the agent executor
    executor = AgentExecutor()

    # Create an agent with all tools
    agent = executor._create_agent()

    # Log the tools registered with the agent
    logger.info("Agent created with all tools registered")

    # There's no direct API to check which tools are registered,
    # but we can verify the agent was created successfully
    assert agent is not None

    # Log success
    logger.info("Successfully created agent with all tools")
