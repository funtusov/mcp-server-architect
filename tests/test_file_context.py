#!/usr/bin/env python3

import os
import tempfile
import pytest
from pathlib import Path

from mcp_server_architect.file_context import FileContextBuilder

class TestFileContextBuilder:
    """Tests for FileContextBuilder class"""
    
    def test_initialization(self):
        """Test FileContextBuilder initialization with valid path"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a builder with a valid directory
            builder = FileContextBuilder(temp_dir)
            assert builder.codebase_path == os.path.abspath(temp_dir)
            assert builder.files_processed == 0
            assert builder.total_context_size == 0
    
    def test_initialization_with_invalid_path(self):
        """Test FileContextBuilder initialization with invalid path"""
        # Test with non-existent path
        with pytest.raises(ValueError, match="Codebase path does not exist"):
            FileContextBuilder("/non/existent/path")
    
    def test_build_context_with_empty_directory(self):
        """Test building context with an empty directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = FileContextBuilder(temp_dir)
            context = builder.build_context()
            # Should include summary text but no file content
            assert "Context includes 0 files out of 0 relevant files found" in context
    
    def test_build_context_with_files(self):
        """Test building context with some files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a Python file
            py_file = os.path.join(temp_dir, "test.py")
            with open(py_file, "w") as f:
                f.write("def test_function():\n    return 'test'")
            
            # Create a README file
            readme_file = os.path.join(temp_dir, "README.md")
            with open(readme_file, "w") as f:
                f.write("# Test Project\nThis is a test.")
            
            # Create an ignored file
            ignored_file = os.path.join(temp_dir, ".env")
            with open(ignored_file, "w") as f:
                f.write("SECRET=test")
            
            builder = FileContextBuilder(temp_dir)
            context = builder.build_context()
            
            # Should find both the Python file and README
            assert "# File: test.py" in context
            assert "def test_function()" in context
            assert "# File: README.md" in context
            assert "# Test Project" in context
            
            # Should not include the ignored .env file
            assert "SECRET=test" not in context