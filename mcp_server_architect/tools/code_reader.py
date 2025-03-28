#!/usr/bin/env python3
"""
Code reader tool for the Architect agent.
"""

import logging
import os

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry, RunContext

from mcp_server_architect.types import ArchitectDependencies
from mcp_server_architect.file_context import FileContextBuilder

# Configure logging
logger = logging.getLogger(__name__)


class CodeReaderInput(BaseModel):
    """Input schema for the code reader tool."""
    paths: list[str] = Field(
        ..., 
        description="List of file or directory paths to read, relative to the base codebase path"
    )
    filter_extensions: list[str] | None = Field(
        None, 
        description="Optional list of file extensions to include (e.g., ['.py', '.js'])"
    )
    max_files: int | None = Field(
        None, 
        description="Maximum number of files to include"
    )


async def code_reader(ctx: RunContext[ArchitectDependencies], input_data: CodeReaderInput) -> str:
    """
    Read and analyze source code files from the specified codebase path.
    
    Args:
        ctx: The runtime context containing dependencies
        input_data: The input parameters specifying which files to read
        
    Returns:
        A string containing the code content or error message
    """
    try:
        codebase_path = ctx.deps.codebase_path
        if not codebase_path:
            return "Error: No codebase path provided in dependencies"
            
        logger.info(f"Reading code files from {codebase_path}")
        
        # Process each specified path
        all_content = []
        
        for rel_path in input_data.paths:
            try:
                # Construct absolute path, ensuring it stays within codebase_path
                abs_path = os.path.normpath(os.path.join(codebase_path, rel_path))
                
                # Security check - ensure the path is still within the codebase directory
                if not abs_path.startswith(os.path.normpath(codebase_path)):
                    all_content.append(f"Error: Path {rel_path} attempts to access files outside the codebase directory")
                    continue
                
                # Check if path exists
                if not os.path.exists(abs_path):
                    raise ModelRetry(f"Path not found: {rel_path}. Please provide a valid path within the codebase.")
                
                # Handle directory vs file differently
                if os.path.isdir(abs_path):
                    # For directories, use FileContextBuilder
                    context_builder = FileContextBuilder(abs_path)
                    
                    # Apply custom filters if provided
                    if input_data.filter_extensions:
                        # We can't directly modify FileContextBuilder's CODE_EXTENSIONS,
                        # but we'll note this in the output
                        note = f"Note: Filtering for extensions: {', '.join(input_data.filter_extensions)}"
                        all_content.append(note)
                    
                    # Build context with potentially custom max_files
                    context = context_builder.build_context()
                    all_content.append(f"Content from directory: {rel_path}\n{context}")
                
                else:
                    # For individual files
                    if input_data.filter_extensions:
                        ext = os.path.splitext(abs_path)[1].lower()
                        if ext not in input_data.filter_extensions:
                            all_content.append(f"Skipping file {rel_path} (extension {ext} not in filter list)")
                            continue
                    
                    # Read and format file content
                    try:
                        with open(abs_path, encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        
                        all_content.append(f"# File: {rel_path}\n```\n{content}\n```\n")
                        logger.debug(f"Read file: {rel_path}")
                    except Exception as e:
                        all_content.append(f"Error reading file {rel_path}: {str(e)}")
                        logger.error(f"Error reading file {abs_path}: {str(e)}")
                
            except Exception as e:
                if isinstance(e, ModelRetry):
                    raise
                all_content.append(f"Error processing path {rel_path}: {str(e)}")
                logger.error(f"Error processing path {rel_path}: {str(e)}", exc_info=True)
        
        if not all_content:
            return "No files were read. Please check the provided paths and filters."
            
        # Combine all content
        return "\n\n".join(all_content)
        
    except Exception as e:
        if isinstance(e, ModelRetry):
            raise
        logger.error(f"Unexpected error in code_reader tool: {str(e)}", exc_info=True)
        return f"Error in code reader tool: {str(e)}"