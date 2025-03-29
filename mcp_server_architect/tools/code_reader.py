#!/usr/bin/env python3
"""
Code reader tool for the Architect agent.
"""

import logging
import os

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry, RunContext

from mcp_server_architect.types import ArchitectDependencies

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_FILES = 50
MAX_CONTEXT_SIZE = 128 * 1024  # 128KB


class CodeReaderInput(BaseModel):
    """Input schema for the code reader tool."""

    paths: list[str] = Field(
        ..., description="List of file or directory paths to read, relative to the base codebase path"
    )
    filter_extensions: list[str] | None = Field(
        None, description="Optional list of file extensions to include (e.g., ['.py', '.js'])"
    )
    max_files: int | None = Field(None, description="Maximum number of files to include")


def is_within_codebase(path: str, codebase_path: str) -> bool:
    """
    Check if a path is within the codebase directory (security check).

    Args:
        path: The absolute path to check
        codebase_path: The base codebase path

    Returns:
        True if the path is within the codebase, False otherwise
    """
    norm_path = os.path.normpath(path)
    norm_codebase = os.path.normpath(codebase_path)

    # Add trailing slash to ensure exact directory match
    if not norm_codebase.endswith(os.sep):
        norm_codebase += os.sep

    return norm_path == norm_codebase or norm_path.startswith(norm_codebase)


def should_include_file(file_path: str, filter_extensions: list[str] | None) -> bool:
    """
    Check if a file should be included based on extension filters.

    Args:
        file_path: The path to the file
        filter_extensions: List of extensions to include, or None to include all

    Returns:
        True if the file should be included, False otherwise
    """
    if not filter_extensions:
        return True

    ext = os.path.splitext(file_path)[1].lower()
    return ext in filter_extensions


def safe_read_file(file_path: str) -> str:
    """
    Safely read a file's content with error handling.

    Args:
        file_path: Path to the file to read

    Returns:
        The file content or an error message
    """
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return f"[Error reading file: {str(e)}]"


def format_file_content(file_path: str, content: str, codebase_path: str) -> str:
    """
    Format a file's content for inclusion in the context.

    Args:
        file_path: Path to the file
        content: The file's content
        codebase_path: The base codebase path for relative path calculation

    Returns:
        The formatted file content with header
    """
    rel_path = os.path.relpath(file_path, codebase_path)
    return f"# File: {rel_path}\n```\n{content}\n```\n\n"


def process_single_path(
    rel_path: str, codebase_path: str, filter_extensions: list[str] | None
) -> tuple[list[str], list[str]]:
    """
    Process a single path (file or directory) and collect matching files.

    Args:
        rel_path: Relative path to process (can be absolute or relative)
        codebase_path: The base codebase path
        filter_extensions: Optional list of file extensions to include

    Returns:
        Tuple of (list of file paths, list of error messages)
    """
    files = []
    errors = []

    try:
        # Determine if the path is already absolute or needs to be joined with codebase path
        if os.path.isabs(rel_path):
            abs_path = os.path.normpath(rel_path)
        else:
            abs_path = os.path.normpath(os.path.join(codebase_path, rel_path))

        # Security check - allow if path equals codebase path or is a subpath
        # For exact match (equal paths), we need special handling
        norm_codebase = os.path.normpath(codebase_path)
        if abs_path != norm_codebase and not is_within_codebase(abs_path, codebase_path):
            errors.append(f"Error: Path {rel_path} attempts to access files outside the codebase directory")
            return files, errors

        # Check if path exists
        if not os.path.exists(abs_path):
            raise ModelRetry(f"Path not found: {rel_path}. Please provide a valid path within the codebase.")

        # Handle directory
        if os.path.isdir(abs_path):
            for root, _, filenames in os.walk(abs_path):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    # Check filter but skip the security check for files within the directory we already verified
                    if should_include_file(file_path, filter_extensions):
                        files.append(file_path)
        # Handle single file
        else:
            if should_include_file(abs_path, filter_extensions):
                files.append(abs_path)

    except Exception as e:
        if isinstance(e, ModelRetry):
            raise
        errors.append(f"Error processing path {rel_path}: {str(e)}")
        logger.error(f"Error processing path {rel_path}: {str(e)}", exc_info=True)

    return files, errors


def collect_files(
    paths: list[str], codebase_path: str, filter_extensions: list[str] | None
) -> tuple[list[str], list[str]]:
    """
    Collect all files from the specified paths.

    Args:
        paths: List of relative paths to process
        codebase_path: The base codebase path
        filter_extensions: Optional list of file extensions to include

    Returns:
        Tuple of (list of file paths, list of error messages)
    """
    all_files = []
    all_errors = []

    for rel_path in paths:
        files, errors = process_single_path(rel_path, codebase_path, filter_extensions)
        all_files.extend(files)
        all_errors.extend(errors)

    return all_files, all_errors


def build_context(
    files: list[str], codebase_path: str, max_files: int = DEFAULT_MAX_FILES, errors: list[str] = None
) -> str:
    """
    Build the context string from the collected files.

    Args:
        files: List of file paths to include
        codebase_path: The base codebase path
        max_files: Maximum number of files to include
        errors: Optional list of error messages to include

    Returns:
        The complete context string
    """
    if not files:
        error_msg = "\n".join(errors) if errors else ""
        return f"No files were found matching the specified criteria. {error_msg}"

    # Apply file limit
    if len(files) > max_files:
        logger.warning(f"Limiting context to {max_files} files out of {len(files)} found")
        files = files[:max_files]

    # Initialize context building
    context_parts = []
    file_count = 0
    total_size = 0

    # Add errors at the beginning if any
    if errors:
        error_section = "# Errors encountered:\n" + "\n".join(errors) + "\n"
        context_parts.append(error_section)
        total_size += len(error_section)

    # Add summary header
    summary = f"# Code Context\nIncluding {len(files)} files from the specified paths.\n\n"
    context_parts.append(summary)
    total_size += len(summary)

    # Process each file
    for file_path in files:
        try:
            # Read and check content
            content = safe_read_file(file_path)
            if not content.strip():
                continue

            # Format content
            formatted_content = format_file_content(file_path, content, codebase_path)

            # Check size limits
            if total_size + len(formatted_content) > MAX_CONTEXT_SIZE:
                logger.warning(f"Context size limit reached at {file_count} files")
                note = f"\n# Note: {len(files) - file_count} more files were found but not included due to size constraints."
                context_parts.append(note)
                break

            # Add to context
            context_parts.append(formatted_content)
            total_size += len(formatted_content)
            file_count += 1

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            error_msg = f"Error reading file {os.path.relpath(file_path, codebase_path)}: {str(e)}\n"
            context_parts.append(error_msg)

    logger.info(f"Built context with {file_count} files, {total_size} characters")
    return "".join(context_parts)


async def code_reader(ctx: RunContext[ArchitectDependencies], input_data: CodeReaderInput) -> str:
    """
    Read and analyze source code files from the specified codebase path, combining them
    into a unified code context for the agent to analyze.

    Args:
        ctx: The runtime context containing dependencies
        input_data: The input parameters specifying which files to read

    Returns:
        A string containing the code context or error message
    """
    try:
        # Get codebase path from dependencies
        codebase_path = ctx.deps.codebase_path
        if not codebase_path:
            return "Error: No codebase path provided in dependencies"

        logger.info(f"Building code context from {len(input_data.paths)} paths in {codebase_path}")

        # Collect files from all specified paths
        all_files, errors = collect_files(input_data.paths, codebase_path, input_data.filter_extensions)

        logger.info(f"Found {len(all_files)} files to include in context")

        # Build the final context
        return build_context(all_files, codebase_path, input_data.max_files or DEFAULT_MAX_FILES, errors)

    except Exception as e:
        if isinstance(e, ModelRetry):
            raise
        logger.error(f"Unexpected error in code_reader tool: {str(e)}", exc_info=True)
        return f"Error in code reader tool: {str(e)}"
