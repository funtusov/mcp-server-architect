#!/usr/bin/env python3
"""
Tests for the LLM tool using litellm with VCR recording.
"""

import os

import pytest
from dotenv import load_dotenv
from pydantic_ai import RunContext

from mcp_server_architect.tools.llm import LLMInput, call_llm_core, llm
from mcp_server_architect.types import ArchitectDependencies

# Load environment variables from .env file
load_dotenv()


@pytest.fixture
def ctx():
    """Create a real RunContext for testing."""
    # Get API keys from environment variables
    api_keys = {
        "gemini": os.environ.get("GEMINI_API_KEY"),
        "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
        "openai": os.environ.get("OPENAI_API_KEY"),
        "openrouter": os.environ.get("OPENROUTER_API_KEY"),
    }

    return RunContext(
        deps=ArchitectDependencies(
            codebase_path="",
            api_keys=api_keys,
        ),
        model="gpt-4o",
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        prompt="test prompt",
        retry=0,
    )


@pytest.mark.asyncio
@pytest.mark.vcr(
    filter_headers=["authorization", "x-goog-api-key", "x-api-key", "anthropic-version"],
    filter_query_parameters=["key", "api_key"],
    record_mode="once",
)
async def test_llm_tool_gpt4o(ctx):
    """Test the LLM tool with GPT-4o model."""
    # Create input data
    input_data = LLMInput(
        prompt="What is 2+2? Respond with just the number.",
        model="gpt4o",
        temperature=0.0,
    )

    # Call the tool
    result = await llm(ctx, input_data)

    # Verify result contains a numeric response and not an error
    assert result is not None
    assert result.strip() != ""
    assert not result.startswith("Error")
    assert any(char.isdigit() for char in result)


@pytest.mark.asyncio
@pytest.mark.vcr(
    filter_headers=["authorization", "x-goog-api-key", "x-api-key", "anthropic-version"],
    filter_query_parameters=["key", "api_key"],
    record_mode="new_episodes",
)
async def test_llm_tool_gemini(ctx):
    """Test the LLM tool with Gemini model."""
    # Create input data
    input_data = LLMInput(
        prompt="What is 3+3? Respond with just the number.",
        model="gemini-2.5-pro",
        temperature=0.0,
    )

    # Call the tool
    result = await llm(ctx, input_data)

    # Verify result contains a numeric response and not an error
    assert result is not None
    assert result.strip() != ""
    assert not result.startswith("Error")
    assert any(char.isdigit() for char in result)


@pytest.mark.asyncio
@pytest.mark.vcr(
    filter_headers=["authorization", "x-goog-api-key", "x-api-key", "anthropic-version"],
    filter_query_parameters=["key", "api_key", "thinking"],
    record_mode="new_episodes",
)
async def test_llm_tool_claude(ctx):
    """Test the LLM tool with Claude 3.7 Sonnet model with thinking enabled."""
    # Create input data
    input_data = LLMInput(
        prompt="What is 4+4? Respond with just the number.",
        model="claude-3.7-sonnet",
        # No temperature - will be forced to 1.0 for Claude with thinking
    )

    # Call the tool
    result = await llm(ctx, input_data)

    # Verify result contains a numeric response and not an error
    assert result is not None
    assert result.strip() != ""
    assert not result.startswith("Error")
    assert any(char.isdigit() for char in result)


@pytest.mark.asyncio
@pytest.mark.vcr(
    filter_headers=["authorization", "x-goog-api-key", "x-api-key", "anthropic-version", "or-organization-id"],
    filter_query_parameters=["key", "api_key"],
    record_mode="once",
)
async def test_llm_tool_deepseek(ctx):
    """Test the LLM tool with DeepSeek model via OpenRouter."""
    # Create input data
    input_data = LLMInput(
        prompt="What is 5+5? Respond with just the number.",
        model="deepseek-v3-0324",
        temperature=0.0,
    )

    # Call the tool
    result = await llm(ctx, input_data)

    # Verify result contains a numeric response and not an error
    assert result is not None
    assert result.strip() != ""
    assert not result.startswith("Error")
    assert any(char.isdigit() for char in result)


@pytest.mark.asyncio
@pytest.mark.vcr(
    filter_headers=["authorization", "x-goog-api-key", "x-api-key", "anthropic-version", "or-organization-id"],
    filter_query_parameters=["key", "api_key"],
    record_mode="once",
)
async def test_llm_tool_error_handling(ctx):
    """Test error handling in the LLM tool with an invalid model."""
    # Create input data with an invalid model ID
    input_data = LLMInput(
        prompt="What is 2+2?",
        model="invalid-model-id",  # This will trigger the fallback logic
        temperature=0.7,
    )

    # Call the tool
    result = await llm(ctx, input_data)

    # The tool should fall back to the default model or show an error
    # In either case, we can at least verify it doesn't crash
    assert result is not None
    assert result.strip() != ""


@pytest.mark.asyncio
@pytest.mark.vcr(
    filter_headers=["authorization", "x-goog-api-key", "x-api-key", "anthropic-version"],
    filter_query_parameters=["key", "api_key"],
    record_mode="once",
)
async def test_core_llm_function():
    """Test the core LLM function that will be used by the MCP tool."""
    # This test simulates the direct use of the core LLM function from MCP
    prompt = "What is 3+5? Respond with just the number."

    # Call the core function directly
    result = await call_llm_core(prompt=prompt, model_id="gemini-2.5-pro", temperature=0.1)

    # Verify result contains a numeric response
    assert result is not None
    assert result.strip() != ""
    assert not result.startswith("Error")
    assert any(char.isdigit() for char in result)

    # Test with default parameters
    result_default = await call_llm_core(prompt=prompt)

    # Verify default parameters work too
    assert result_default is not None
    assert result_default.strip() != ""
    assert not result_default.startswith("Error")
