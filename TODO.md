# Architect AI - MCP Server

This project implements a Model Context Protocol (MCP) server that acts as an AI Software Architect.

It exposes a tool (`ArchitectAI::generate_prd`) that takes:
1.  A detailed description of a programming task.
2.  A local path to a codebase directory.

The server then:
1.  Scans the specified directory for code and configuration files.
2.  Formats the file contents along with the task description into a prompt.
3.  Sends the prompt to Google's Gemini Pro model.
4.  Returns the LLM's response, which should be a Product Requirements Document (PRD) or a High-Level Design Document suitable for guiding an AI coding assistant.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Install `uv`:**
    If you don't have `uv` installed, follow the instructions at [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv). Often, this is:
    ```bash
    # macOS / Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Windows
    irm https://astral.sh/uv/install.ps1 | iex
    # Or via pip (if pip is already available)
    pip install uv
    ```

3.  **Create and Activate Virtual Environment:**
    ```bash
    uv venv
    source .venv/bin/activate  # Linux/macOS
    # .venv\Scripts\activate  # Windows PowerShell
    # source .venv/Scripts/activate # Git Bash on Windows
    ```

4.  **Install Dependencies:**
    Use `uv` to install the required packages:
    ```bash
    uv pip install "mcp-sdk[cli]" google-generativeai python-dotenv
    ```
    *   `mcp-sdk[cli]`: The Model Context Protocol SDK and command-line tools.
    *   `google-generativeai`: Google's SDK for Gemini.
    *   `python-dotenv`: To load environment variables from a `.env` file.

5.  **Configure API Key:**
    *   Copy the example environment file: `cp .env.example .env`
    *   Edit the `.env` file and add your Google API Key obtained from [Google AI Studio](https://aistudio.google.com/app/apikey):
        ```dotenv
        GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
        # Optionally change GEMINI_MODEL if needed
        ```

## Running the Server

There are two main ways to run the server locally using `uv` and the `mcp` command-line tool:

1.  **Development Mode (Recommended for Testing):**
    This mode uses the MCP Inspector, allowing you to easily view server logs and manually trigger tools/resources. The `--with-editable .` flag ensures changes to your Python files (`main.py`, `file_context.py`) are picked up automatically.
    ```bash
    uv run mcp dev main.py --with-editable .
    ```
    Open the URL provided by the command (usually `http://localhost:8787`) in your browser to access the MCP Inspector.

2.  **Direct Execution Mode:**
    This runs the server directly without the Inspector UI.
    ```bash
    uv run mcp run main.py
    ```
    The server will listen for MCP connections (typically via stdio or SSE, depending on how clients connect).

## Integrating with Claude Desktop

Once the server is running (usually in direct execution mode for stable use, although `mcp install` might handle running it), you can make its tools available within Claude Desktop:

1.  **Install the Server into Claude:**
    Run the following command in your terminal (make sure your virtual environment is active):
    ```bash
    uv run mcp install main.py --name ArchitectAI
    ```
    *   `uv run mcp install`: Uses `uv` to execute the `mcp install` command from the SDK.
    *   `main.py`: Specifies the entry point file for your server.
    *   `--name ArchitectAI`: Gives the server a recognizable name within Claude.

2.  **Using the Tool in Claude:**
    After installation, the `ArchitectAI::generate_prd` tool should become available to Claude. When you want Claude to use it, you would typically instruct it in your conversation, providing the necessary arguments. For example:

    ```
    @ArchitectAI please generate a PRD for refactoring the authentication.
    Task Description: "Refactor the authentication logic currently in app/main.py. Create a dedicated auth module (e.g., app/auth.py) with functions for user login, registration, and token verification. Update the main app to use this new module. Ensure password hashing uses bcrypt."
    Codebase Path: "/path/to/your/local/project/sample_codebase"
    ```

    Claude's MCP client integration will handle calling the tool with the specified `task_description` and `codebase_path` arguments. Make sure the `codebase_path` is accessible from where the MCP server process is running.

    *(Note: The exact syntax for invoking tools within Claude might vary slightly depending on the Claude Desktop version and its specific MCP integration.)*
