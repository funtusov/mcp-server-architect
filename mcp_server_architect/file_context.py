#!/usr/bin/env python3
"""
File context builder for analyzing codebases.
"""

import fnmatch
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Common code file extensions to look for
CODE_EXTENSIONS = {
    # Programming languages
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".kt",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rb",
    ".php",
    ".swift",
    ".scala",
    ".rs",
    ".dart",
    ".lua",
    ".groovy",
    ".pl",
    ".pm",
    ".r",
    ".clj",
    ".ex",
    ".exs",
    ".elm",
    ".erl",
    ".fs",
    ".fsx",
    ".hs",
    # Web
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sass",
    ".less",
    # Data/Config
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".toml",
    ".ini",
    ".cfg",
    # Shell/Scripts
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".bat",
    ".cmd",
    # Documentation
    ".md",
    ".rst",
    ".txt",
}

# Files/directories to ignore
DEFAULT_IGNORE_PATTERNS = [
    # Hidden files/dirs
    ".*",
    ".*/.*",
    # Build artifacts and dependencies
    "node_modules/*",
    "venv/*",
    ".venv/*",
    "env/*",
    ".env/*",
    "__pycache__/*",
    "build/*",
    "dist/*",
    "target/*",
    "out/*",
    "bin/*",
    "obj/*",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.dll",
    "*.class",
    # Large data files
    "*.csv",
    "*.tsv",
    "*.parquet",
    "*.avro",
    "*.pb",
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.gif",
    "*.bmp",
    "*.tiff",
    "*.ico",
    "*.mp3",
    "*.mp4",
    "*.avi",
    "*.mov",
    "*.wav",
    "*.flac",
    "*.pdf",
    "*.doc",
    "*.docx",
    "*.ppt",
    "*.pptx",
    "*.xls",
    "*.xlsx",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.tgz",
    "*.rar",
    "*.7z",
    "*.log",
    "*.log.*",
    "*.data",
    "*.db",
    "*.sqlite",
    "*.sqlite3",
]

# Maximum file size to include (in bytes)
MAX_FILE_SIZE = 1024 * 1024  # 1MB

# Maximum number of files to include
MAX_FILES = 50

# Maximum total context size (in characters)
MAX_CONTEXT_SIZE = 128 * 1024  # 128KB


class FileContextBuilder:
    """
    Builds context from files in a codebase directory for use in generating PRDs.
    """

    def __init__(self, codebase_path: str):
        """
        Initialize the FileContextBuilder.

        Args:
            codebase_path (str): Path to the codebase directory
        """
        self.codebase_path = os.path.abspath(os.path.expanduser(codebase_path))
        self.files_processed = 0
        self.total_context_size = 0

        # Check if the path exists and is a directory
        if not os.path.exists(self.codebase_path):
            raise ValueError(f"Codebase path does not exist: {self.codebase_path}")
        if not os.path.isdir(self.codebase_path):
            raise ValueError(f"Codebase path is not a directory: {self.codebase_path}")

    def build_context(self) -> str:
        """
        Build a context string from relevant files in the codebase.

        Returns:
            str: The combined context from relevant files
        """
        logger.info(f"Building context from codebase at: {self.codebase_path}")

        # Find relevant files
        relevant_files = self._find_relevant_files()
        logger.info(f"Found {len(relevant_files)} relevant files")

        # Read and format file contents
        context_parts = []
        for file_path in relevant_files:
            try:
                # Check if we've reached the maximum files limit
                if self.files_processed >= MAX_FILES:
                    logger.warning(f"Reached maximum file limit of {MAX_FILES}")
                    break

                # Read file content and format it
                relative_path = os.path.relpath(file_path, self.codebase_path)
                file_content = self._read_file(file_path)

                # Skip if file is empty
                if not file_content.strip():
                    continue

                # Format the file content with path as header
                formatted_content = f"# File: {relative_path}\n```\n{file_content}\n```\n\n"

                # Check if adding this file would exceed the maximum context size
                if self.total_context_size + len(formatted_content) > MAX_CONTEXT_SIZE:
                    logger.warning(f"Reached maximum context size of {MAX_CONTEXT_SIZE} characters")
                    break

                context_parts.append(formatted_content)
                self.total_context_size += len(formatted_content)
                self.files_processed += 1

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)

        # Combine all parts into a single context string
        context = "\n".join(context_parts)

        # Add summary of what was included
        summary = (
            f"Context includes {self.files_processed} files out of {len(relevant_files)} relevant files found.\n\n"
        )

        logger.info(f"Built context with {self.files_processed} files totaling {self.total_context_size} characters")
        return summary + context

    def _find_relevant_files(self) -> list[str]:
        """
        Find relevant files in the codebase directory.

        Returns:
            List[str]: List of absolute paths to relevant files
        """
        relevant_files = []

        # Helper function to check if a path should be ignored
        def should_ignore(path: str) -> bool:
            rel_path = os.path.relpath(path, self.codebase_path)
            return any(fnmatch.fnmatch(rel_path, pattern) for pattern in DEFAULT_IGNORE_PATTERNS)

        # Walk through the directory tree
        for root, dirs, files in os.walk(self.codebase_path):
            # Filter out directories that should be ignored (in-place)
            dirs[:] = [d for d in dirs if not should_ignore(os.path.join(root, d))]

            # Filter and add relevant files
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()

                # Skip files that should be ignored
                if should_ignore(file_path):
                    continue

                # Skip files that are too large
                if os.path.getsize(file_path) > MAX_FILE_SIZE:
                    logger.debug(f"Skipping large file: {file_path}")
                    continue

                # Include if it has a code file extension
                if file_ext in CODE_EXTENSIONS:
                    relevant_files.append(file_path)

        # Sort files by importance/recency (e.g., prefer README files, main entry points)
        return self._prioritize_files(relevant_files)

    def _prioritize_files(self, file_paths: list[str]) -> list[str]:
        """
        Prioritize files based on importance for context building.

        Args:
            file_paths (List[str]): List of file paths to prioritize

        Returns:
            List[str]: Prioritized list of file paths
        """
        # Define priority categories
        readme_files = []
        config_files = []
        main_files = []
        regular_files = []

        for file_path in file_paths:
            file_name = os.path.basename(file_path).lower()
            os.path.relpath(file_path, self.codebase_path)

            # README files
            if file_name.startswith("readme"):
                readme_files.append(file_path)
            # Configuration files
            elif any(file_name.endswith(ext) for ext in [".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"]):
                config_files.append(file_path)
            # Main entry point files
            elif file_name in ["main.py", "app.py", "index.js", "server.js", "application.java", "program.cs"]:
                main_files.append(file_path)
            # Regular code files
            else:
                regular_files.append(file_path)

        # Combine in priority order
        return readme_files + config_files + main_files + regular_files

    def _read_file(self, file_path: str) -> str:
        """
        Read the content of a file safely.

        Args:
            file_path (str): Path to the file to read

        Returns:
            str: The content of the file
        """
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return f"[Error reading file: {str(e)}]"
