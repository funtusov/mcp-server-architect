#!/usr/bin/env python3
"""
Tests for the web search tool using Exa AI.
"""

import asyncio
import os

import pytest

from mcp_server_architect.tools.web_search import WebSearchInput
from mcp_server_architect.tools.web_search import web_search as web_search_fn
from mcp_server_architect.types import ArchitectDependencies


class MockRunContext:
    """Mock RunContext class for testing."""
    
    def __init__(self, api_key):
        """Initialize with just the necessary attributes."""
        self.deps = ArchitectDependencies(
            codebase_path="",
            api_keys={"web_search": api_key}
        )


@pytest.mark.vcr(
    filter_headers=["authorization", "x-api-key"],
    filter_query_parameters=["key", "api_key"],
    record_mode="new_episodes"
)
def test_web_search_basic():
    """Test basic web search functionality."""
    # Check for API key
    exa_key = os.environ.get("EXA_API_KEY")
    if not exa_key:
        raise RuntimeError("EXA_API_KEY environment variable is required for web search tests")
    
    # Create a simple mock context
    context = MockRunContext(exa_key)
    
    # Create search input
    search_input = WebSearchInput(
        query="What is the Model Context Protocol",
        num_results=3
    )
    
    # Execute the search using event loop
    result = asyncio.run(web_search_fn(context, search_input))
    
    # Verify the results
    assert result is not None
    assert result.strip() != ""
    assert "Search Results" in result
    assert "Model Context Protocol" in result or "MCP" in result
    
    # Verify result contains expected sections
    assert "URL:" in result
    
    # Check if we got the expected number of results (at least 1)
    result_sections = result.count("##")
    assert result_sections > 0, "Expected at least one search result"


@pytest.mark.vcr(
    filter_headers=["authorization", "x-api-key"],
    filter_query_parameters=["key", "api_key"],
    record_mode="new_episodes"
)
def test_web_search_technical():
    """Test web search with technical query."""
    # Check for API key
    exa_key = os.environ.get("EXA_API_KEY")
    if not exa_key:
        raise RuntimeError("EXA_API_KEY environment variable is required for web search tests")
    
    # Create a simple mock context
    context = MockRunContext(exa_key)
    
    # Create search input for technical query
    search_input = WebSearchInput(
        query="PydanticAI agent pattern implementation example",
        num_results=2
    )
    
    # Execute the search using event loop
    result = asyncio.run(web_search_fn(context, search_input))
    
    # Verify the results
    assert result is not None
    assert result.strip() != ""
    assert "Search Results" in result
    
    # Check for relevant content
    assert any(term in result.lower() for term in ["pydantic", "agent", "ai", "example"])