# mcp-server-architect

A Model Context Protocol server that acts as an AI Software Architect. It analyzes codebases and generates Product Requirements Documents (PRDs) or High-Level Design Documents based on task descriptions.

## Features

- Analyze local codebase directories to understand project structure
- Generate comprehensive PRDs or design documents for new features
- Integrates with Claude Code via MCP
- Uses Google's Gemini Pro model for content generation
- Easy to install and run with `uvx mcp-server-architect`

## Prerequisites

- Python 3.10 or higher
- Google API key for Gemini Pro (get one from [Google AI Studio](https://aistudio.google.com/app/apikey))

## Installation

### Quick Installation with uv (Recommended)

The simplest way to install and use the server is with `uv` package manager:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# No installation needed - run directly with uvx (one-liner)
GEMINI_API_KEY=your_api_key_here uvx mcp-server-architect
```

### Installation with pip

You can also install the package from PyPI:

```bash
pip install mcp-server-architect
```

After installation, you can run it as a command:

```bash
GEMINI_API_KEY=your_api_key_here mcp-server-architect
```

### API Key Requirements

This server requires a Gemini API key for accessing the Google Gemini model. You can obtain one from [Google AI Studio](https://aistudio.google.com/app/apikey). The API key can be provided in multiple ways:

1. As an environment variable inline: `GEMINI_API_KEY=your_key mcp-server-architect`
2. By setting it before running: `export GEMINI_API_KEY=your_key`
3. Through a .env file in the current directory

### Development Installation

If you're developing or modifying the code:

1. **Clone the Repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```

2. **Setup Development Environment:**
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e ".[dev]"
   ```

3. **Run in Development Mode:**
   ```bash
   GEMINI_API_KEY=your_api_key_here python -m mcp_server_architect
   ```

4. **Run with MCP Inspector for Development:**
   ```bash
   GEMINI_API_KEY=your_api_key_here npx @modelcontextprotocol/inspector python -m mcp dev --with-editable . mcp_server_architect/__main__.py
   ```

## Running and Using the Server

### Direct Execution with uvx

The easiest way to run the server is with `uvx`, passing your Gemini API key:

```bash
# As a one-liner (recommended)
GEMINI_API_KEY=your_api_key_here uvx mcp-server-architect

# Or export it first
export GEMINI_API_KEY="your_api_key_here"
uvx mcp-server-architect
```

### Using with MCP Inspector

To debug or test the server with the MCP Inspector:

```bash
GEMINI_API_KEY=your_api_key_here npx @modelcontextprotocol/inspector uvx mcp-server-architect
```

This will open an inspector interface (usually at http://localhost:8787) that allows you to test the server's tools interactively.

### Adding to Claude Code

Claude Code supports MCP servers in various scopes. Here's how to add the ArchitectAI server with your Gemini API key:

```bash
# Local scope (only available to you in the current project)
claude mcp add architect -- GEMINI_API_KEY=your_api_key_here uvx mcp-server-architect

# Project scope (shared with everyone via .mcp.json)
claude mcp add architect -s project -- GEMINI_API_KEY=your_api_key_here uvx mcp-server-architect

# User scope (available to you across all projects)
claude mcp add architect -s user -- GEMINI_API_KEY=your_api_key_here uvx mcp-server-architect
```

> **Important:** Replace `your_api_key_here` with your actual Google API key for Gemini

#### Understanding MCP Server Scopes

Claude Code provides three different scopes for MCP servers:

- **Local** (default): Available only to you in the current project
- **Project**: Stored in a `.mcp.json` file that can be committed to version control and shared with your team
- **User**: Available to you across all your projects

For team collaboration, the **Project** scope is recommended as it allows everyone on the team to access the same MCP servers without individual setup.

#### Storing API Keys Securely

For security, you may want to store your API key in a more secure way. You can:

1. **Use a .env file (in project scope):**
   ```bash
   # Create a .env file (don't commit this!)
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   
   # Then use the dotenv command to load it
   claude mcp add architect -- sh -c "source .env && exec uvx mcp-server-architect"
   ```

2. **Use your OS's secure credential storage:**
   - On macOS, you can store it in Keychain and retrieve it with a script
   - Add the script reference in your command

#### Verifying Installation

After installation, you can verify the server is registered with Claude:

```bash
# List all configured servers
claude mcp list

# Get details for the architect server
claude mcp get architect
```

### Running Tests

To run the test suite:

```bash
# Using uv
uv run pytest

# Using pip
python -m pytest
```

## MCP Resources and Tools

This MCP server exposes the following resources and tools:

### Tools

- **`ArchitectAI::generate_prd`**: Generates a Product Requirements Document based on codebase analysis
  - Parameters:
    - `task_description` (required): Detailed description of the programming task or feature to implement
    - `codebase_path` (required): Local file path to the codebase directory to analyze

### Usage Examples

After installation, you can use the tool in Claude Code by prompting:

```
@ArchitectAI please generate a PRD for creating a new feature.
Task Description: "Create a user profile page that displays user information and activity history, with edit functionality."
Codebase Path: "/path/to/your/local/project"
```

Example with more specific technical details:

```
@ArchitectAI generate a PRD for a new feature.
Task Description: "Implement JWT authentication in a Flask application, with login, registration, and token refresh endpoints. Add middleware for protected routes and handle token expiration gracefully."
Codebase Path: "/Users/username/projects/my-flask-app"
```

You can also create a custom slash command for easier access:

1. Create a commands directory in your project:
   ```bash
   mkdir -p .claude/commands
   ```

2. Create a command file for ArchitectAI:
   ```bash
   echo "Generate a PRD for the following task:\n\nTask Description: \"$ARGUMENTS\"\nCodebase Path: \"`pwd`\"" > .claude/commands/prd.md
   ```

3. Use it in Claude Code:
   ```
   /project:prd Implement a new user authentication system
   ```

## Building and Publishing

To build and publish the package to PyPI using uv:

1. **Build the package:**
   ```bash
   uv build --no-sources
   ```
   This creates distribution packages in the `dist/` directory.

2. **Publish to TestPyPI** (optional but recommended):
   ```bash
   # Set your TestPyPI token
   export UV_PUBLISH_TOKEN=your_testpypi_token
   
   # Publish to TestPyPI
   uv publish --publish-url https://test.pypi.org/legacy/
   ```

3. **Publish to PyPI:**
   ```bash
   # Set your PyPI token
   export UV_PUBLISH_TOKEN=your_pypi_token
   
   # Publish to PyPI
   uv publish
   ```

### Release Steps Summary

Here's a summary of all steps to prepare and release a new version:

1. Update version following semantic versioning (major.minor.patch) in:
   - `pyproject.toml`
   - `mcp_server_architect/version.py`
   - `mcp_server_architect/__init__.py`

2. Make sure tests pass:
   ```bash
   uv run pytest
   ```

3. Build the package:
   ```bash
   uv build --no-sources
   ```

4. Test the package locally:
   ```bash
   # Create a temporary directory
   mkdir -p /tmp/test-architect
   cd /tmp/test-architect
   
   # Test installing from the built package
   uv run --with-pin /path/to/your/dist/mcp_server_architect-*.whl --no-project -- python -c "from mcp_server_architect import __version__; print(__version__)"
   ```

5. Publish to PyPI:
   ```bash
   uv publish
   ```

6. Verify the installation:
   ```bash
   # In a fresh environment
   uvx mcp-server-architect --version
   ```

## License

[MIT](LICENSE)
