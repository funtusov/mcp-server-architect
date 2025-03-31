# mcp-server-architect

A Model Context Protocol server that acts as an AI Software Architect. It analyzes codebases to generate Product Requirements Documents (PRDs) and provides reasoning assistance for complex coding tasks using a powerful agent-based architecture.

## Features

- **Multi-Model Architecture**: Uses OpenAI's GPT-4o for primary agent tasks with access to specialized tools
- **Intelligent Codebase Analysis**: Builds comprehensive code context from project files for architectural understanding
- **Agent-Based Design**: Uses a smart agent that autonomously decides which tools to employ for each task
- **Tool-Based Processing**: Equipped with specialized tools for code reading, web searches, and targeted LLM queries
- **Comprehensive PRD Generation**: Creates detailed product requirement documents with architectural insights
- **Advanced Reasoning**: Helps developers solve complex coding challenges with step-by-step reasoning
- **Logfire Instrumentation**: Built-in monitoring and debugging of agent activity with detailed telemetry
- **MCP Integration**: Seamlessly connects with Claude Code via the Model Context Protocol
- **Simple Deployment**: Quick to install and run with `uvx mcp-server-architect`

## How It Works

The Architect MCP Server implements a sophisticated agent-based architecture that mimics how a human software architect would approach complex design tasks:

1. **Agent Loop**: When a request is received (either for PRD generation or reasoning assistance), a primary GPT-4o based agent evaluates the task and orchestrates the solution process.

2. **Tool-Based Architecture**: The agent has access to specialized tools:
   - **Code Reader**: Analyzes source code files and combines them into a coherent context representation
   - **Web Search**: Uses Exa AI to find relevant technical information online
   - **LLM Tool**: Makes targeted calls to specialized language models for specific sub-tasks

3. **Autonomous Decision Making**: The agent determines which tools to use, when to use them, and how to synthesize their outputs to produce the final result.

4. **Contextual Awareness**: For PRD generation, the system builds a deep understanding of your codebase structure, dependencies, and design patterns before making recommendations.

5. **Flexible Response Generation**: All outputs are formatted in clear, structured markdown for easy integration into your workflow.

### Component Architecture

The system follows a modular design with the following key components:

```
┌─────────────────────────────────────────────────────────────────┐
│                       MCP Server Interface                      │
│                    (mcp_server_architect/__main__.py)           │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Architect Core                            │
│                    (mcp_server_architect/core.py)               │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Agent Executor                            │
│                (mcp_server_architect/agents/executor.py)        │
└───┬─────────────────────┬────────────────────────┬──────────────┘
    │                     │                        │
    ▼                     ▼                        ▼
┌───────────────┐    ┌─────────────┐         ┌─────────────┐
│ LLM Models    │    │    Tools    │         │ Dependencies│
│ OpenAI GPT-4o │    │ code_reader │         │ArchitectDeps│
│ Gemini 2.5    │    │ web_search  │         └─────────────┘
└───────────────┘    │ llm         │
                     └─────────────┘
```

#### Component Flow:

1. **MCP Server Interface**: Entry point that exposes the services via Model Context Protocol
   - Registers tools (`generate_prd` and `think`)
   - Handles incoming requests and routes them to the core

2. **Architect Core**: Central component that coordinates operations
   - Manages agent creation and execution
   - Implements the public API (generate_prd, think)
   - Handles errors and logging

3. **Agent Executor**: Creates and configures agents with appropriate models and tools
   - Selects models based on task (OpenAI or Gemini)
   - Uses direct model initialization for OpenAI models
   - Registers tools with the agent
   - Provides methods for running the agent for different tasks

4. **LLM Models**:
   - OpenAI GPT-4o for main agent loop (default for general tasks)
   - Gemini 2.5 for specific tasks (PRD generation and thinking)

5. **Tools**:
   - `code_reader`: Analyzes source code files from a codebase
   - `web_search`: Searches the web for relevant information
   - `llm`: Makes targeted LLM calls for specific sub-tasks

6. **Dependencies**:
   - `ArchitectDependencies`: Provides codebase path and API keys to tools

#### Data Flow:

1. User request → MCP Server Interface
2. Interface routes request → Architect Core
3. Core calls appropriate method on Agent Executor
4. Agent Executor creates and configures agent
5. Agent executes with tools, accessing models as needed
6. Results flow back through the same chain
7. Formatted response returned to user

See the [CHANGELOG](CHANGELOG.md) for details on the latest improvements.

## Prerequisites

- Python 3.10 or higher
- OpenAI API key for GPT-4o (get one from [OpenAI Platform](https://platform.openai.com/api-keys))
- Google API key for Gemini Pro (get one from [Google AI Studio](https://aistudio.google.com/app/apikey)) 
- Exa API key for web search capabilities (get one from [Exa AI](https://exa.ai))
- Logfire API key for monitoring (optional, get one from [Logfire](https://logfire.sh/))

The system will prioritize using OpenAI's models for the main agent tasks, while using Google Gemini for specific tool operations. Both AI model API keys are recommended for optimal performance. The Logfire API key is optional but provides valuable telemetry for monitoring and debugging agent activity.

## Installation

### Quick Installation with uv (Recommended)

The simplest way to install and use the server is with `uv` package manager:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# No installation needed - run directly with uvx (one-liner)
env GEMINI_API_KEY=your_api_key_here uvx mcp-server-architect
```

### Installation with pip

You can also install the package from PyPI:

```bash
pip install mcp-server-architect
```

After installation, you can run it as a command:

```bash
env GEMINI_API_KEY=your_api_key_here mcp-server-architect
```

### API Key Requirements

This server requires a Gemini API key for accessing the Google Gemini model. You can obtain one from [Google AI Studio](https://aistudio.google.com/app/apikey). The API key can be provided in multiple ways:

1. As an environment variable with the env prefix: `env GEMINI_API_KEY=your_key mcp-server-architect`
2. Through a .env file in the current directory with `GEMINI_API_KEY=your_key`

**Note:** Setting the environment variable with `export` before running may not work reliably. The `env` command prefix is the recommended approach.

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
   env GEMINI_API_KEY=your_api_key_here python -m mcp_server_architect
   ```

4. **Run with MCP Inspector for Development:**
   ```bash
   env GEMINI_API_KEY=your_api_key_here npx @modelcontextprotocol/inspector python -m mcp dev --with-editable . mcp_server_architect/__main__.py
   ```

## Running and Using the Server

### Direct Execution with uvx

The easiest way to run the server is with `uvx`, passing your Gemini API key:

```bash
# As a one-liner (recommended)
env GEMINI_API_KEY=your_api_key_here EXA_API_KEY=your_exa_key_here uvx mcp-server-architect

# Alternatively, use a .env file in the current directory
# with GEMINI_API_KEY and EXA_API_KEY environment variables
```

### Using with MCP Inspector

To debug or test the server with the MCP Inspector:

```bash
env GEMINI_API_KEY=your_api_key_here EXA_API_KEY=your_exa_key_here npx @modelcontextprotocol/inspector uvx mcp-server-architect
```

This will open an inspector interface (usually at http://localhost:8787) that allows you to test the server's tools interactively.

### Adding to Claude Code

Claude Code supports MCP servers in various scopes. Here's how to add the Architect server with your Gemini API key:

```bash
# Local scope (only available to you in the current project)
claude mcp add architect -- env GEMINI_API_KEY=your_api_key_here EXA_API_KEY=your_exa_key_here uvx mcp-server-architect

# Project scope (shared with everyone via .mcp.json)
claude mcp add architect -s project -- env GEMINI_API_KEY=your_api_key_here EXA_API_KEY=your_exa_key_here uvx mcp-server-architect

# User scope (available to you across all projects)
claude mcp add architect -s user -- env GEMINI_API_KEY=your_api_key_here EXA_API_KEY=your_exa_key_here uvx mcp-server-architect
```

> **Important:** Replace `your_api_key_here` with your actual Google API key for Gemini and `your_exa_key_here` with your Exa API key for web search

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
   echo "EXA_API_KEY=your_exa_key_here" >> .env
   
   # Add to Claude Code with the env prefix
   claude mcp add architect -- env GEMINI_API_KEY=your_api_key_here EXA_API_KEY=your_exa_key_here uvx mcp-server-architect
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

The tests use pytest-recording to record HTTP interactions with APIs. By default, tests will use previously recorded responses. To update the recordings:

```bash
# Force rewrite of API recordings
pytest --record-mode=rewrite
```

For more details about testing, see [tests/README.md](tests/README.md).

## MCP Resources and Tools

This MCP server exposes the following resources and tools:

### Tools

- **`Architect::generate_prd`**: Generates a Product Requirements Document based on codebase analysis
  - Parameters:
    - `task_description` (required): Detailed description of the programming task or feature to implement
    - `codebase_path` (required): Local file path to the codebase directory to analyze

- **`Architect::think`**: Provides reasoning assistance for a stuck LLM on a coding task
  - Parameters:
    - `request` (required): Detailed description of the coding task/issue and relevant code snippets

### Usage Examples

#### Generate PRD Example

After installation, you can use the PRD tool in Claude Code by prompting:

```
@Architect please generate a PRD for creating a new feature.
Task Description: "Create a user profile page that displays user information and activity history, with edit functionality."
Codebase Path: "/path/to/your/local/project"
```

Example with more specific technical details:

```
@Architect generate a PRD for a new feature.
Task Description: "Implement JWT authentication in a Flask application, with login, registration, and token refresh endpoints. Add middleware for protected routes and handle token expiration gracefully."
Codebase Path: "/Users/username/projects/my-flask-app"
```

#### Reasoning Assistance Example

When you're stuck on a coding task, use the thinking tool to get detailed reasoning:

```
@Architect I need help thinking through a coding problem.
I'm trying to implement a function that reverses a linked list but I'm stuck on handling the edge cases.
Here's my code:
```python
def reverse_linked_list(head):
    if not head or not head.next:
        return head
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev
```
What edge cases am I missing? Is my implementation correct?
```

You can also create custom slash commands for easier access:

1. Create a commands directory in your project:
   ```bash
   mkdir -p .claude/commands
   ```

2. Create command files for Architect tools:
   ```bash
   # PRD generation command
   echo "Generate a PRD for the following task:\n\nTask Description: \"$ARGUMENTS\"\nCodebase Path: \"`pwd`\"" > .claude/commands/prd.md
   
   # Thinking assistance command
   echo "I need help thinking through this coding problem:\n\n$ARGUMENTS" > .claude/commands/think.md
   ```

3. Use them in Claude Code:
   ```
   # For PRD generation
   /project:prd Implement a new user authentication system
   
   # For reasoning assistance
   /project:think I'm trying to optimize this recursive function but hitting a stack overflow...
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
