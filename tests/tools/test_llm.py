#!/usr/bin/env python3
"""
Tests for the LLM tool.
"""

from unittest.mock import MagicMock, patch

import pytest

from mcp_server_architect.tools.llm import LLMInput, llm
from mcp_server_architect.types import ArchitectDependencies


class TestLLMTool:
    """Tests for the LLM tool functions."""

    @pytest.mark.asyncio
    @patch("mcp_server_architect.tools.llm.Agent")
    async def test_llm_tool_basic(self, mock_agent):
        """Test the basic functionality of the LLM tool."""
        # Setup mock agent
        mock_agent_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.data = "This is a test response"
        
        # Create a proper async mock for the run method
        async def mock_run(*args, **kwargs):
            return mock_response
            
        mock_agent_instance.run = mock_run
        mock_agent.return_value = mock_agent_instance
        
        # Create context and input
        mock_ctx = MagicMock()
        mock_ctx.deps = ArchitectDependencies(
            codebase_path="",
            api_keys={}
        )
        
        input_data = LLMInput(
            prompt="Test prompt",
            temperature=0.7
        )
        
        # Call the tool
        result = await llm(mock_ctx, input_data)
        
        # Verify result
        assert result == "This is a test response"
        mock_agent.assert_called_once()

    @pytest.mark.asyncio
    @patch("mcp_server_architect.tools.llm.Agent")
    async def test_llm_tool_with_custom_model(self, mock_agent):
        """Test the LLM tool with custom model selection."""
        # Setup mock agent
        mock_agent_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.data = "Custom model response"
        
        # Keep track of what was passed to run
        run_args = {}
        
        # Create a proper async mock for the run method
        async def mock_run(*args, **kwargs):
            run_args.update(kwargs)  # Store the kwargs for later inspection
            return mock_response
            
        mock_agent_instance.run = mock_run
        mock_agent.return_value = mock_agent_instance
        
        # Create context and input
        mock_ctx = MagicMock()
        mock_ctx.deps = ArchitectDependencies(
            codebase_path="",
            api_keys={}
        )
        
        input_data = LLMInput(
            prompt="Test prompt",
            model="gpt4o",
            temperature=0.5,
            max_tokens=100
        )
        
        # Call the tool
        result = await llm(mock_ctx, input_data)
        
        # Verify result
        assert result == "Custom model response"
        mock_agent.assert_called_once()
        
        # Verify generation options were passed correctly
        assert run_args["generation_options"]["temperature"] == 0.5
        assert run_args["generation_options"]["max_tokens"] == 100

    @pytest.mark.asyncio
    @patch("mcp_server_architect.tools.llm.Agent")
    async def test_llm_tool_error_handling(self, mock_agent):
        """Test error handling in the LLM tool."""
        # Setup mock agent to raise an exception
        mock_agent_instance = MagicMock()
        
        # Create a proper async mock that raises an exception
        async def mock_run_error(*args, **kwargs):
            raise Exception("Test error")
            
        mock_agent_instance.run = mock_run_error
        mock_agent.return_value = mock_agent_instance
        
        # Create context and input
        mock_ctx = MagicMock()
        mock_ctx.deps = ArchitectDependencies(
            codebase_path="",
            api_keys={}
        )
        
        input_data = LLMInput(
            prompt="Test prompt"
        )
        
        # Call the tool
        result = await llm(mock_ctx, input_data)
        
        # Verify error handling
        assert "Error in LLM tool" in result
        assert "Test error" in result