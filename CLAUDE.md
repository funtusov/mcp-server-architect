# Claude Guidelines for architect-mcp

## Commands
- Setup: `uv add -e .` (install in dev mode)
- Dependencies: `uv add google-generativeai python-dotenv` (required packages)
- Build: `uv build --no-sources`
- Run: `uvx mcp-server-architect`
- Lint: `ruff check .`
- Format: `ruff format .`
- Fix lint issues: `ruff check --fix .`

## Pre-publishing Checklist
Before publishing a new version:
1. Update version numbers in:
   - `mcp_server_architect/version.py`
   - `mcp_server_architect/__init__.py`
   - `pyproject.toml`
2. Run `./test_build.sh` to verify everything works
3. Run `uv build --no-sources`
4. Publish with `uv publish`

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