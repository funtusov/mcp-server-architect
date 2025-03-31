# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-03-28

### Added
- Agent-based architecture using PydanticAI
- Web search tool using Exa API for retrieving information
- Code reader tool for analyzing source code files
- Updated LLM tool for generating content using Gemini models
- Support for agent loop with multiple steps before returning a response

### Changed
- Refactored `Architect` class to use AgentExecutor for running agent loops
- Renamed `AgentManager` to `AgentExecutor` for better clarity on its role
- Consolidated separate agents into a single unified agent with task-specific prompts
- Updated default Gemini model to `gemini-2.5-pro-exp-03-25`
- Improved error handling and logging across components
- Added codebase_path parameter to think tool for better reasoning with codebase context

### Fixed
- Various linting issues and code style improvements

## [0.1.7] - 2025-03-27

### Added
- Retry mechanism for Gemini API calls
- Updated API usage patterns

## [0.1.6] - 2025-03-26

### Changed
- Updated version in `__init__.py`

## [0.1.5] - 2025-03-25

### Changed
- Renamed ArchitectAI to Architect
- Updated author information

## [0.1.0] - 2025-03-20

### Added
- Initial release
- Support for generating PRDs based on codebase analysis
- Support for providing reasoning assistance
- Integration with FastMCP