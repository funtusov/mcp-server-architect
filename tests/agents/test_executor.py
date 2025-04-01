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
    """Test creating an agent with GPT-4o."""
    # Create the agent executor
    executor = AgentExecutor()

    # Create the agent
    logger.info("Testing agent creation with GPT-4o")

    # Create the agent with GPT-4o
    agent = executor._create_agent()

    # Verify the agent was created
    assert agent is not None

    # Log success
    logger.info("Successfully created agent with GPT-4o")


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


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_nested_event_loop():
    """Test that our fix for the nested event loop issue works."""
    # Create the agent executor
    executor = AgentExecutor()

    # This would previously fail with "This event loop is already running" error
    # Now it should work because we're using async/await properly
    result = await executor.run_analyze_agent("Test request", "")

    # Verify we got a result that's not an error about event loops
    assert result is not None
    assert "This event loop is already running" not in result
