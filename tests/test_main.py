#!/usr/bin/env python3

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

# We need to patch the genai configuration to avoid requiring an API key during tests
with patch('google.generativeai.configure'):
    from mcp_server_architect.core import ArchitectAI


class TestArchitectAI:
    """Tests for ArchitectAI class"""
    
    def setup_method(self):
        """Set up for tests"""
        self.architect = ArchitectAI()
    
    def test_create_prompt(self):
        """Test the prompt creation method"""
        task_description = "Implement user authentication"
        code_context = "def login(): pass"
        
        prompt = self.architect._create_prompt(task_description, code_context)
        
        # Check that the task description and code context are included in the prompt
        assert task_description in prompt
        assert code_context in prompt
        assert "You are an expert software architect" in prompt
        assert "Format your response in markdown" in prompt
    
    def test_process_response_text_attribute(self):
        """Test processing a response with text attribute"""
        mock_response = MagicMock()
        mock_response.text = "Test response"
        
        result = self.architect._process_response(mock_response)
        assert result == "Test response"
    
    def test_process_response_parts_attribute(self):
        """Test processing a response with parts attribute"""
        mock_part1 = MagicMock()
        mock_part1.text = "Part 1"
        mock_part2 = MagicMock()
        mock_part2.text = "Part 2"
        
        mock_response = MagicMock()
        mock_response.text = None  # No text attribute
        mock_response.parts = [mock_part1, mock_part2]
        
        result = self.architect._process_response(mock_response)
        assert result == "Part 1Part 2"
    
    def test_process_response_fallback(self):
        """Test processing a response without text or parts attributes"""
        mock_response = MagicMock()
        mock_response.text = None
        mock_response.parts = None
        mock_response.__str__.return_value = "String representation"
        
        result = self.architect._process_response(mock_response)
        assert result == "String representation"
    
    @patch('mcp_server_architect.core.FileContextBuilder')
    @patch('google.generativeai.GenerativeModel')
    def test_generate_prd(self, mock_generative_model_class, mock_file_context_builder_class):
        """Test the generate_prd method with mocked dependencies"""
        # Setup mocks
        mock_context_builder = MagicMock()
        mock_context_builder.build_context.return_value = "Mocked code context"
        mock_file_context_builder_class.return_value = mock_context_builder
        
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Generated PRD content"
        mock_model.generate_content.return_value = mock_response
        mock_generative_model_class.return_value = mock_model
        
        # Call the method
        result = self.architect.generate_prd(
            task_description="Implement user authentication",
            codebase_path="/fake/path"
        )
        
        # Verify the result
        assert result == "Generated PRD content"
        
        # Verify the mocks were called correctly
        mock_file_context_builder_class.assert_called_once_with("/fake/path")
        mock_context_builder.build_context.assert_called_once()
        mock_generative_model_class.assert_called_once()
        mock_model.generate_content.assert_called_once()
        
    @patch('mcp_server_architect.core.FileContextBuilder')
    def test_generate_prd_error_handling(self, mock_file_context_builder_class):
        """Test error handling in the generate_prd method"""
        # Setup mock to raise an exception
        mock_context_builder = MagicMock()
        mock_context_builder.build_context.side_effect = Exception("Test error")
        mock_file_context_builder_class.return_value = mock_context_builder
        
        # Call the method
        result = self.architect.generate_prd(
            task_description="Implement user authentication",
            codebase_path="/fake/path"
        )
        
        # Verify the result contains the error message
        assert "Error generating PRD: Test error" in result