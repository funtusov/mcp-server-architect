#!/usr/bin/env python3
"""
Tests for the code_reader tool.
"""

import os
from unittest.mock import patch

import pytest
from pydantic_ai import ModelRetry, RunContext

from mcp_server_architect.tools.code_reader import (
    CodeReaderInput,
    build_context,
    code_reader,
    collect_files,
    format_file_content,
    is_within_codebase,
    process_single_path,
    safe_read_file,
    should_include_file,
)
from mcp_server_architect.types import ArchitectDependencies


class TestCodeReader:
    """Tests for the code reader tool functions."""

    @pytest.fixture
    def test_data_path(self):
        """Return the path to the test data directory."""
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data", "code_reader")

    @pytest.fixture
    def ctx(self, test_data_path):
        """Create a real RunContext for testing."""
        return RunContext(
            deps=ArchitectDependencies(
                codebase_path=test_data_path,
                api_keys={},
            ),
            model="gpt-4o",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            prompt="test prompt",
            retry=0,
        )

    def test_is_within_codebase(self):
        """Test the is_within_codebase function."""
        codebase_path = "/path/to/codebase"
        assert is_within_codebase("/path/to/codebase/file.py", codebase_path) is True
        assert is_within_codebase("/path/to/codebase/subfolder/file.py", codebase_path) is True
        assert is_within_codebase("/path/to/other/file.py", codebase_path) is False
        assert is_within_codebase("/path/to/codebases/file.py", codebase_path) is False

    def test_should_include_file(self):
        """Test the should_include_file function."""
        # Test with no filter
        assert should_include_file("file.py", None) is True
        assert should_include_file("file.js", None) is True

        # Test with filter
        assert should_include_file("file.py", [".py"]) is True
        assert should_include_file("file.js", [".py"]) is False
        assert should_include_file("file.py", [".js", ".py"]) is True

    def test_safe_read_file(self, test_data_path):
        """Test the safe_read_file function."""
        # Test with valid file
        file_path = os.path.join(test_data_path, "file1.py")
        content = safe_read_file(file_path)
        assert "Test file 1" in content

        # Test with non-existent file
        content = safe_read_file(os.path.join(test_data_path, "nonexistent.py"))
        assert "Error reading file" in content

    def test_format_file_content(self, test_data_path):
        """Test the format_file_content function."""
        file_path = os.path.join(test_data_path, "file1.py")
        content = "Sample content"
        formatted = format_file_content(file_path, content, os.path.dirname(test_data_path))

        assert "# File:" in formatted
        assert "```" in formatted
        assert "Sample content" in formatted

    def test_process_single_path(self, test_data_path):
        """Test the process_single_path function."""
        # Test with single file
        files, errors = process_single_path("file1.py", test_data_path, None)
        assert len(files) == 1
        assert os.path.basename(files[0]) == "file1.py"
        assert not errors

        # Test with directory using absolute path
        files, errors = process_single_path(test_data_path, test_data_path, None)
        assert len(files) >= 2  # Should find at least file1.py and file2.py
        assert not errors

        # Test with filters
        files, errors = process_single_path(test_data_path, test_data_path, [".py"])
        assert all(f.endswith(".py") for f in files)

        # Test with non-existent path
        with pytest.raises(ModelRetry) as excinfo:
            process_single_path("nonexistent", test_data_path, None)
        assert "Path not found" in str(excinfo.value)

        # Test with path outside codebase
        files, errors = process_single_path("../..", test_data_path, None)
        assert not files
        assert len(errors) == 1
        assert "outside the codebase directory" in errors[0]

    def test_collect_files(self, test_data_path):
        """Test the collect_files function."""
        # Test with multiple paths
        files, errors = collect_files(["file1.py", "file2.py"], test_data_path, None)
        assert len(files) == 2
        assert not errors

        # Test with directory and filter
        files, errors = collect_files([test_data_path], test_data_path, [".py"])
        assert all(f.endswith(".py") for f in files)
        assert not errors

        # Test with mixture of valid and invalid paths
        with pytest.raises(ModelRetry) as excinfo:
            files, errors = collect_files(["file1.py", "nonexistent.py"], test_data_path, None)
        assert "Path not found" in str(excinfo.value)

    def test_build_context(self, test_data_path):
        """Test the build_context function."""
        # Get file paths
        file1_path = os.path.join(test_data_path, "file1.py")
        file2_path = os.path.join(test_data_path, "file2.py")

        # Test with valid files
        context = build_context([file1_path, file2_path], test_data_path)
        assert "# Code Context" in context
        assert "Including 2 files" in context
        assert "Test file 1" in context
        assert "Test file 2" in context

        # Test with max files
        context = build_context([file1_path, file2_path], test_data_path, max_files=1)
        assert "Including 1 files" in context
        assert "Test file 1" in context
        assert "Test file 2" not in context

        # Test with errors
        context = build_context([file1_path], test_data_path, errors=["Test error"])
        assert "# Errors encountered:" in context
        assert "Test error" in context

        # Test with no files
        context = build_context([], test_data_path)
        assert "No files were found" in context

    @pytest.mark.asyncio
    async def test_code_reader_tool(self, test_data_path, ctx):
        """Test the code_reader tool function."""
        # Create input data
        input_data = CodeReaderInput(paths=[test_data_path], filter_extensions=[".py"])

        # Call the tool
        result = await code_reader(ctx, input_data)

        # Verify result
        assert "# Code Context" in result
        assert "Including" in result
        assert "Test file 1" in result
        assert "Test file 2" in result

        # Test with no codebase path
        empty_ctx = RunContext(
            deps=ArchitectDependencies(codebase_path="", api_keys={}),
            model="gpt-4o",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            prompt="test prompt",
            retry=0,
        )
        result = await code_reader(empty_ctx, input_data)
        assert "Error: No codebase path provided" in result

        # Test with specific files
        input_data = CodeReaderInput(paths=["file1.py"], max_files=1)
        result = await code_reader(ctx, input_data)
        assert "Test file 1" in result
        assert "function1" in result

    @pytest.mark.asyncio
    async def test_code_reader_error_handling(self, test_data_path, ctx):
        """Test error handling in the code_reader tool."""

        # Test with non-existent path
        with pytest.raises(ModelRetry) as excinfo:
            input_data = CodeReaderInput(paths=["nonexistent.py"])
            await code_reader(ctx, input_data)
        assert "Path not found" in str(excinfo.value)

        # Test with exception in collect_files
        with patch("mcp_server_architect.tools.code_reader.collect_files") as mock_collect:
            mock_collect.side_effect = Exception("Test exception")

            input_data = CodeReaderInput(paths=["file1.py"])
            result = await code_reader(ctx, input_data)
            assert "Error in code reader tool" in result
            assert "Test exception" in result


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
