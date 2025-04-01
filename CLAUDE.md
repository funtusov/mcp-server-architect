# Claude Guidelines for architect-mcp

## Commands
- Setup: `uv add -e .` (install in dev mode)
- Dependencies: `uv add google-genai python-dotenv` (required packages)
- Build: `uv build --no-sources`
- Run: `uvx mcp-server-architect`
- Lint: `ruff check .`
- Format: `ruff format .`
- Fix lint issues: `ruff check --fix .`

## Publishing a New Version
For a complete guide on building, testing, and publishing new versions, see [PUBLISH.md](PUBLISH.md) for detailed step-by-step instructions.

## Testing
- Run the pytest integration tests when developing and before publishing
- For normal development, use linting to check code quality: `ruff check .`

### Integration Tests Best Practices
- Use pytest-recording for integration tests to record live API responses
- Never use mocks in integration tests; they create brittle tests that don't catch real issues
- Run tests with recorded cassettes:
  ```bash
  uv run pytest tests/ -v
  ```
- To rewrite all cassettes:
  ```bash
  uv run pytest tests/ --record-mode=all
  ```
- Remember to check cassettes into source control but filter sensitive data

## Committing Changes
When committing changes:
1. Use a succinct one-line commit message
2. Don't include "Generated with Claude Code" or "Co-Authored-By" lines
3. Run linter checks before committing
4. Always add all modified files including lock files for a clean state (`git add .`)
5. If certain files should not be committed, explicitly gitignore them

## UV Cheatsheet
- Add dependency: `uv add <package>` or `uv add <package> --dev`
- Remove dependency: `uv remove <package>` or `uv remove <package> --dev`
- Run in venv: `uv run <command>` (e.g. `uv run pytest`)
- Sync environment: `uv sync`
- Update lockfile: `uv lock`
- Install tool: `uv tool install <tool>` (e.g. `uv tool install ruff`)

## Code Style
- Line length: 120 characters (configured in pyproject.toml)
- Python 3.10+ compatible
- Document functions with docstrings (triple quotes)
- File structure: shebang line, docstring, imports, constants, classes/functions
- Imports: standard library first, then third-party, then local modules
- Error handling: use try/except blocks with specific exceptions and logging
- Naming: snake_case for variables/functions, PascalCase for classes
- Logging: use the module-level logger defined at the top of each file

## Architecture
- MCP server with FastMCP integration
- Multi-model architecture: GPT-4o for main agent, Gemini for specific tasks
- Agent-based processing with tool orchestration
- Context-building from codebase files
- Logfire instrumentation for monitoring and debugging
- Clean error handling with detailed logging

## Environment Variables
- `OPENAI_API_KEY`: OpenAI API key for GPT-4o (primary agent)
- `GEMINI_API_KEY`: Google AI Studio API key for Gemini (backup and tool operations)
- `EXA_API_KEY`: Exa AI API key for web search capabilities
- `LOGFIRE_API_KEY`: Logfire API key for agent instrumentation

## Project Documentation
When implementing features, refer to documentation located in `.llm/docs/` directory:

Available docs:
- PydanticAI
  - Intro: `.llm/docs/pydantic_ai/intro.md`
  - Agents: `.llm/docs/pydantic_ai/agents.md`
  - Tools: `.llm/docs/pydantic_ai/tools.md`
  - Models: `.llm/docs/pydantic_ai/models.md`
  - Index: `.llm/docs/pydantic_ai/index.md`

Always read these docs when working with PydanticAI or implementing agent-based functionality to follow project conventions and best practices.