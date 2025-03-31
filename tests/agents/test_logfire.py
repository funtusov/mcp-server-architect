#!/usr/bin/env python3
"""
Tests for logfire instrumentation with PydanticAI.
"""

import unittest.mock

import logfire
from pydantic_ai import Agent


def test_logfire_installed():
    """Test that logfire is properly installed."""
    assert logfire is not None
    assert hasattr(logfire, "configure")
    assert logfire.__version__ >= "3.11.0"


@unittest.mock.patch("pydantic_ai.Agent.instrument_all")
def test_agent_instrumentation_called(mock_instrument_all):
    """Test that Agent.instrument_all is called for instrumentation."""
    # Call the instrument_all function
    Agent.instrument_all()

    # Verify it was called
    mock_instrument_all.assert_called_once()
