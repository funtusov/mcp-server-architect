#!/usr/bin/env python3
"""
Agent Executor module that handles the configuration and execution of PydanticAI agents.
"""

import logging
import os

import logfire
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from mcp_server_architect.models import get_model_string
from mcp_server_architect.tools.code_reader import code_reader
from mcp_server_architect.tools.llm import llm
from mcp_server_architect.tools.web_search import web_search
from mcp_server_architect.types import ArchitectDependencies

# Configure logging
logger = logging.getLogger(__name__)

# Configure logfire and instrument pydantic-ai agents
try:
    # Attempt to configure logfire
    logfire.configure(service_name="architect-mcp", ignore_no_config=True)

    # Use the built-in PydanticAI instrumentation
    Agent.instrument_all()
    logger.info("Logfire configured and PydanticAI agents instrumented successfully")
except Exception as e:
    logger.warning(f"Failed to configure logfire and instrumentation: {str(e)}")


class AgentExecutor:
    """
    Agent executor that creates and runs a unified agent with different task-specific prompts.
    Uses a consistent system prompt while varying execution prompts by task type.
    """

    def __init__(self):
        """Initialize the AgentExecutor."""
        self.api_keys = self._gather_api_keys()
        logger.info("AgentExecutor initialized with API keys for %s services", len(self.api_keys))

    def _gather_api_keys(self) -> dict[str, str]:
        """Gather API keys from environment variables."""
        api_keys = {}

        # OpenAI API key (required)
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            api_keys["openai"] = openai_key
        else:
            logger.warning("OPENAI_API_KEY environment variable is not set")

        return api_keys

    def _create_agent(self, task: str = None) -> Agent:
        """
        Create a PydanticAI agent with the architect system prompt.
        Uses direct model initialization for OpenAI models.

        Args:
            task: Optional task name to select the appropriate model

        Returns:
            A configured PydanticAI Agent
        """
        # Set API keys in environment for pydantic-ai to use
        if "gemini" in self.api_keys:
            os.environ["GEMINI_API_KEY"] = self.api_keys["gemini"]
        if "openai" in self.api_keys:
            os.environ["OPENAI_API_KEY"] = self.api_keys["openai"]

        # Define the unified system prompt for the architect agent
        system_prompt = """
        You are an expert software architect and technical lead with deep expertise in software design and development.
        
        You can provide several types of assistance:
        1. Create Product Requirements Documents (PRDs) or High-Level Design Documents
        2. Analyze code and architecture
        3. Provide deep reasoning assistance and solutions for complex coding problems
        
        Use the tools strategically to gather all the information you need:
        - code_reader: Read source code files from the codebase to understand the architecture
        - web_search: Find relevant technical information online
        - llm: Use external LLM assistance when needed
        
        Format your responses in markdown. Be concise but thorough.
        """

        # Get the appropriate model string for this task
        model_string = get_model_string(task)
        logger.info(f"Creating agent with model: {model_string}")

        # For OpenAI models, use direct initialization to avoid compatibility issues
        if model_string.startswith("openai:"):
            # Extract just the model name from the string
            model_name = model_string.split(":", 1)[1]
            logger.info(f"Using direct model initialization for OpenAI model: {model_name}")

            # Create OpenAI model instance directly
            model = OpenAIModel(model_name=model_name, provider="openai")

            # Create agent with explicit model instance
            agent = Agent(
                model,
                deps_type=ArchitectDependencies,
                system_prompt=system_prompt,
            )
        else:
            # For non-OpenAI models (like Gemini), use the standard string format
            logger.info(f"Using string-based initialization for model: {model_string}")
            agent = Agent(
                model_string,
                deps_type=ArchitectDependencies,
                system_prompt=system_prompt,
            )

        # Register all tools
        agent.tool(code_reader)
        agent.tool(web_search)
        agent.tool(llm)

        return agent

    async def run_prd_agent(self, task_description: str, codebase_path: str) -> str:
        """
        Run the agent to generate a PRD using a generic agent pattern.
        The agent will use tools as needed to gather information and generate the PRD.

        Args:
            task_description: Detailed description of the programming task
            codebase_path: Path to the local codebase directory

        Returns:
            The generated PRD text
        """
        logger.info(f"Generating PRD for task: {task_description[:50]}...")
        logger.info(f"Using codebase path: {codebase_path}")

        # Create the agent with the task-specific model (using "generate_prd" task)
        agent = self._create_agent(task="generate_prd")

        try:
            # Prepare dependencies
            deps = ArchitectDependencies(codebase_path=codebase_path, api_keys=self.api_keys)

            # Create a task-specific prompt for PRD generation
            prd_prompt = """
            Create a Product Requirements Document (PRD) or High-Level Design Document based on the following task.
            
            IMPORTANT: Before beginning analysis, use the code_reader tool to examine the codebase structure
            and understand the existing architecture.
            
            Your PRD should include:
            1. Overview of the requested feature/task
            2. Technical requirements and constraints
            3. Proposed architecture/design
            4. Implementation plan with specific files to modify
            5. Potential challenges and mitigations
            
            Task details: {task_description}
            """

            # Format the prompt with the task description
            prompt = prd_prompt.format(task_description=task_description)

            # The agent is already instrumented via Agent.instrument_all()
            # Use run() instead of run_sync() to avoid nested event loops
            result = await agent.run(prompt, deps=deps)

            # Extract and return the response
            return result.data

        except Exception as e:
            logger.error(f"Error in PRD generation agent: {str(e)}", exc_info=True)
            return f"Error generating PRD: {str(e)}"

    async def run_analyze_agent(self, request: str, codebase_path: str) -> str:
        """
        Run the agent to analyze code and respond to user queries.

        Args:
            request: Description of what to analyze
            codebase_path: Path to the codebase

        Returns:
            Analysis result
        """
        logger.info(f"Analyzing codebase for request: {request[:50]}...")
        logger.info(f"Using codebase path: {codebase_path}")

        # Create the unified agent
        agent = self._create_agent(task="think")

        try:
            # Prepare dependencies
            deps = ArchitectDependencies(codebase_path=codebase_path, api_keys=self.api_keys)

            # Create a task-specific prompt for reasoning assistance
            analysis_prompt = """
            Analyze the following coding problem or question and provide reasoning assistance.
            
            If a codebase path is provided, use the code_reader tool to examine relevant files.
            
            In your response:
            1. Break down the problem step by step
            2. Identify potential solutions or approaches
            3. Explain your reasoning thoroughly
            4. Include relevant code examples where helpful
            
            Problem/Question: {request}
            """

            # Format the prompt with the request
            prompt = analysis_prompt.format(request=request)

            # The agent is already instrumented via Agent.instrument_all()
            # Use run() instead of run_sync() to avoid nested event loops
            result = await agent.run(prompt, deps=deps)

            # Extract and return the response
            return result.data

        except Exception as e:
            logger.error(f"Error in code analysis agent: {str(e)}", exc_info=True)
            return f"Error analyzing code: {str(e)}"
