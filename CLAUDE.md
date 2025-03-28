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
- Only run `test_build.sh` before publishing OR when explicitly requested
- The test script makes API calls to Gemini which:
  - Incurs API costs
  - Is slow (30+ seconds)
  - Should not be run routinely during development
- For normal development, use linting to check code quality: `ruff check .`

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
- Gemini API for generative AI capabilities
- Context-building from codebase files
- Clean error handling with detailed logging