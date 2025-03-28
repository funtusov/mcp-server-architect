#!/usr/bin/env python3
"""
Test PydanticAI Agent functionality to debug our issue.
"""

import logging
from dataclasses import dataclass

import pytest
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class TestDependencies:
    """Test dependencies for agent."""
    api_key: str


class TestInput(BaseModel):
    """Test input model."""
    message: str = Field(..., description="Message to echo")


async def test_tool(ctx: RunContext[TestDependencies], input_data: TestInput) -> str:
    """
    Simple echo tool for testing.
    
    Args:
        ctx: The runtime context
        input_data: The input data
        
    Returns:
        The message from the input
    """
    return f"Echo: {input_data.message}"


def test_agent_api():
    """Test the PydanticAI Agent API to understand how it works."""
    # Create test dependencies
    deps = TestDependencies(api_key="test-key")
    
    # Create a system prompt
    system_prompt = "You are a test agent."
    
    # Create the agent with proper provider:model format
    agent = Agent(
        "test",  # Use the test model built into pydantic-ai
        deps_type=TestDependencies,
        system_prompt=system_prompt
    )
    
    # Check what methods are available on the agent
    logger.info(f"Agent methods: {dir(agent)}")
    
    # Test different ways to add tools
    try:
        # Method 1: add_tool()
        if hasattr(agent, "add_tool"):
            agent.add_tool(test_tool)
            logger.info("Added tool using add_tool()")
        # Method 2: register_tool()
        elif hasattr(agent, "register_tool"):
            agent.register_tool(test_tool)
            logger.info("Added tool using register_tool()")
        else:
            logger.error("Neither add_tool() nor register_tool() methods exist")
            logger.info(f"Available methods: {[m for m in dir(agent) if not m.startswith('_')]}")
    except Exception as e:
        logger.error(f"Error adding tool: {str(e)}")
    
    # Print the agent configuration
    logger.info(f"Agent configuration: {agent}")
    
    # Assert that we were able to add the tool somehow
    assert True  # We just want to see the logs