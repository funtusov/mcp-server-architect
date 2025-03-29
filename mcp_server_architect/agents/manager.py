#!/usr/bin/env python3
"""
Agent Manager module that handles the configuration and execution of PydanticAI agents.
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
    logfire.configure(
        service_name="architect-mcp",
        ignore_no_config=True
    )
    
    # Use the built-in PydanticAI instrumentation
    Agent.instrument_all()
    logger.info("Logfire configured and PydanticAI agents instrumented successfully")
except Exception as e:
    logger.warning(f"Failed to configure logfire and instrumentation: {str(e)}")


class AgentManager:
    """
    Simplified AgentManager that uses only GPT-4o and the code_reader tool.
    """

    def __init__(self):
        """Initialize the AgentManager."""
        self.api_keys = self._gather_api_keys()
        logger.info("AgentManager initialized with API keys for %s services", 
                   len(self.api_keys))

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

    def _create_agent(self, system_prompt: str, task: str = None) -> Agent:
        """
        Create a PydanticAI agent with the provided system prompt.
        Uses direct model initialization for OpenAI models.
        
        Args:
            system_prompt: The system prompt for the agent
            task: Optional task name to select the appropriate model
            
        Returns:
            A configured PydanticAI Agent
        """
        # Set API keys in environment for pydantic-ai to use
        if "gemini" in self.api_keys:
            os.environ["GEMINI_API_KEY"] = self.api_keys["gemini"]
        if "openai" in self.api_keys:
            os.environ["OPENAI_API_KEY"] = self.api_keys["openai"]
        
        # Get the appropriate model string for this task
        model_string = get_model_string(task)
        logger.info(f"Creating agent with model: {model_string}")
        
        # For OpenAI models, use direct initialization to avoid compatibility issues
        if model_string.startswith("openai:"):
            # Extract just the model name from the string
            model_name = model_string.split(":", 1)[1]
            logger.info(f"Using direct model initialization for OpenAI model: {model_name}")
            
            # Create OpenAI model instance directly
            model = OpenAIModel(
                model_name=model_name,
                provider="openai"
            )
            
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

    def run_prd_agent(self, task_description: str, codebase_path: str) -> str:
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
        
        # Define the system prompt for PRD generation
        system_prompt = """
        You are an expert software architect and technical lead.
        
        Your task is to create a Product Requirements Document (PRD) or High-Level Design Document based on the 
        user's request and any code context you gather using your tools.
        
        IMPORTANT: Before beginning any analysis, use the code_reader tool to examine the codebase structure
        and understand the existing architecture.
        
        Your PRD should include:
        1. Overview of the requested feature/task
        2. Technical requirements and constraints
        3. Proposed architecture/design
        4. Implementation plan with specific files to modify
        5. Potential challenges and mitigations
        
        You have access to the following tool:
        - code_reader: Read source code files from the codebase to understand the architecture
        
        Format your response in markdown. Be concise but comprehensive.
        Use your tools strategically to gather all the information you need.
        """
        
        # Create the agent with the task-specific model (using "generate_prd" task)
        agent = self._create_agent(system_prompt, task="generate_prd")
        
        try:
            # Prepare dependencies
            deps = ArchitectDependencies(
                codebase_path=codebase_path,
                api_keys=self.api_keys
            )
            
            # Run the agent with the task description
            prompt = f"Generate a PRD for the following task: {task_description}"
            
            # The agent is already instrumented via Agent.instrument_all()
            result = agent.run_sync(prompt, deps=deps)
            
            # Extract and return the response
            return result.data
            
        except Exception as e:
            logger.error(f"Error in PRD generation agent: {str(e)}", exc_info=True)
            return f"Error generating PRD: {str(e)}"

    def run_analyze_agent(self, request: str, codebase_path: str) -> str:
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
        
        # Define system prompt for code analysis
        system_prompt = """
        You are an expert software developer with deep expertise in code analysis.
        
        Your task is to analyze the codebase and answer the user's query.
        Use the code_reader tool to examine relevant files in the codebase.
        
        In your response:
        1. Explain what you found in the code
        2. Answer the specific query thoroughly
        3. Include code snippets where helpful
        
        Format your response in markdown. Be concise but thorough.
        """
        
        # Create agent
        agent = self._create_agent(system_prompt)
        
        try:
            # Prepare dependencies
            deps = ArchitectDependencies(
                codebase_path=codebase_path,
                api_keys=self.api_keys
            )
            
            # Run the agent with the request
            # The agent is already instrumented via Agent.instrument_all()
            result = agent.run_sync(request, deps=deps)
            
            # Extract and return the response
            return result.data
            
        except Exception as e:
            logger.error(f"Error in code analysis agent: {str(e)}", exc_info=True)
            return f"Error analyzing code: {str(e)}"