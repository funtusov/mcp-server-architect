#!/usr/bin/env python3
"""
Integration tests for the Architect MCP Server.
"""

import os

import pytest

from mcp_server_architect.core import Architect

# Directory for test data
TEST_CODEBASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_codebase")


def scrub_api_keys(response):
    """Remove API keys from the recorded responses."""
    if isinstance(response, dict) and "body" in response and "access_token" in response["body"]:
        response["body"]["access_token"] = "[FILTERED]"  # noqa: S105
    return response


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment with test codebase."""
    # We rely on conftest.py to ensure the API keys are loaded
    # This fixture only sets up the test codebase
    
    # Make sure test directories exist
    os.makedirs(TEST_CODEBASE_PATH, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cassettes"), exist_ok=True)
    
    # Create a simple Python file in the test codebase
    with open(os.path.join(TEST_CODEBASE_PATH, "main.py"), "w") as f:
        f.write('''
def add(a, b):
    """Add two numbers."""
    return a + b

def subtract(a, b):
    """Subtract b from a."""
    return a - b
''')
    
    yield
    
    # Cleanup is optional - we keep the test files for inspection


@pytest.mark.vcr(
    filter_headers=["authorization", "x-goog-api-key"],
    filter_query_parameters=["key", "api_key"],
    before_record_response=scrub_api_keys,
    record_mode="new_episodes"
)
def test_generate_prd():
    """Test the generate_prd function."""
    # Initialize the Architect
    architect = Architect()
    
    # Define test parameters
    task_description = "Create a simple calculator app with basic operations"
    
    # Call the function we want to test
    result = architect.generate_prd(task_description, TEST_CODEBASE_PATH)
    
    # Assertions - check for expected patterns in the response, not exact matches
    assert result is not None
    assert result.strip() != ""
    assert "calculator" in result.lower() or "app" in result.lower()
    assert any(keyword in result.lower() for keyword in ["overview", "requirements", "implementation"])


@pytest.mark.vcr(
    filter_headers=["authorization", "x-goog-api-key"],
    filter_query_parameters=["key", "api_key"],
    before_record_response=scrub_api_keys,
    record_mode="new_episodes"
)
def test_think():
    """Test the think function."""
    # Initialize the Architect
    architect = Architect()
    
    # Define test request
    request = "How should I implement error handling in my calculator app?"
    
    # Call the function
    result = architect.think(request)
    
    # Assertions
    assert result is not None
    assert result.strip() != ""
    assert any(keyword in result.lower() for keyword in [
        "error", "handling", "try", "except", "validate", "calculator"
    ])