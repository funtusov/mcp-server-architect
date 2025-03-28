#!/usr/bin/env python3
"""
Agent Manager module that handles the configuration and execution of PydanticAI agents.
"""

import logging
import os

from pydantic_ai import Agent

from mcp_server_architect.models import get_model_string
from mcp_server_architect.tools.code_reader import code_reader
from mcp_server_architect.tools.llm import llm
from mcp_server_architect.tools.web_search import web_search
from mcp_server_architect.types import ArchitectDependencies

# Configure logging
logger = logging.getLogger(__name__)


class AgentManager:
    """
    Manages the configuration and execution of PydanticAI agents for the Architect service.
    Handles agent initialization, tool registration, and provides methods to execute
    different agent tasks (generate_prd, think).
    """

    def __init__(self):
        """Initialize the AgentManager."""
        self.api_keys = self._gather_api_keys()
        logger.info("AgentManager initialized with API keys for %s services", 
                   len(self.api_keys))

    def _gather_api_keys(self) -> dict[str, str]:
        """Gather API keys from environment variables."""
        api_keys = {}
        
        # Gemini API key
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            api_keys["gemini"] = gemini_key
        else:
            logger.warning("GEMINI_API_KEY environment variable is not set")
        
        # OpenAI API key
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            api_keys["openai"] = openai_key
        else:
            logger.warning("OPENAI_API_KEY environment variable is not set")
            
        # Web Search API key (if available) - using EXA_API_KEY
        web_search_key = os.getenv("EXA_API_KEY")
        if web_search_key:
            api_keys["web_search"] = web_search_key
            logger.info("EXA_API_KEY environment check: Set")
        else:
            logger.warning("EXA_API_KEY environment variable is not set. Web search will not work.")
        
        return api_keys

    def _create_agent(self, system_prompt: str, task: str = None) -> Agent:
        """
        Create a PydanticAI agent with the provided system prompt.
        
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
        
        # Create agent with the model string
        agent = Agent(
            model_string,
            deps_type=ArchitectDependencies,
            system_prompt=system_prompt,
        )
        
        # Register tools
        agent.tool(web_search)
        agent.tool(code_reader)
        agent.tool(llm)
        
        return agent

    def run_prd_agent(self, task_description: str, codebase_path: str) -> str:
        """
        Run the agent to generate a PRD.
        
        Args:
            task_description: Detailed description of the programming task
            codebase_path: Path to the local codebase directory
            
        Returns:
            The generated PRD text
        """
        logger.info(f"Generating PRD for task: {task_description[:50]}...")
        logger.info(f"Using codebase path: {codebase_path}")
        
        # Create system prompt for PRD generation
        system_prompt = """
        You are an expert software architect and technical lead.
        
        Your task is to create a Product Requirements Document (PRD) or High-Level Design Document based on the 
        user's request and any code context you gather using your tools.
        
        Your PRD should include:
        1. Overview of the requested feature/task
        2. Technical requirements and constraints
        3. Proposed architecture/design
        4. Implementation plan with specific files to modify
        5. Potential challenges and mitigations
        
        Format your response in markdown. Be concise but comprehensive.
        You have access to tools to help with your task. Use them as needed.
        """
        
        # Create the agent with appropriate tools, specifying the task for model selection
        agent = self._create_agent(system_prompt, task="generate_prd")
        
        try:
            # Prepare dependencies
            deps = ArchitectDependencies(
                codebase_path=codebase_path,
                api_keys=self.api_keys
            )
            
            # Run the agent with the task description
            prompt = f"Generate a PRD for the following task: {task_description}"
            result = agent.run_sync(prompt, deps=deps)
            
            # Extract and return the response
            return result.data
            
        except Exception as e:
            logger.error(f"Error in PRD generation agent: {str(e)}", exc_info=True)
            return f"Error generating PRD: {str(e)}"

    def run_think_agent(self, request: str) -> str:
        """
        Run the agent to provide reasoning assistance.
        
        Args:
            request: Detailed description of the coding task/issue and relevant code snippets
            
        Returns:
            Reasoning guidance and potential solutions
        """
        logger.info(f"Providing reasoning assistance for request: {request[:50]}...")
        
        # Create system prompt for reasoning assistance
        system_prompt = """
        You are an expert software developer with deep expertise in code analysis and problem-solving.
        
        You need to help with a coding task that might be complex or challenging. Analyze the request and provide
        your reasoning, insights, and potential solutions.
        
        In your response:
        1. Identify the core problem or challenge
        2. Break down the problem into manageable steps
        3. Provide specific coding approaches or patterns to resolve the issue
        4. Suggest alternative solutions when appropriate
        5. Explain your reasoning clearly
        
        Format your response in markdown. Be concise but thorough. Focus on practical, implementable solutions.
        You have access to tools to help with your task. Use them as needed.
        """
        
        # Create the agent with appropriate tools, specifying the task for model selection
        agent = self._create_agent(system_prompt, task="think")
        
        try:
            # Prepare dependencies - we don't have a codebase path for thinking, 
            # but we might still need API keys for web search
            deps = ArchitectDependencies(
                codebase_path="",  # Empty as we don't have a specific codebase path
                api_keys=self.api_keys
            )
            
            # Run the agent with the request
            result = agent.run_sync(request, deps=deps)
            
            # Extract and return the response
            return result.data
            
        except Exception as e:
            logger.error(f"Error in reasoning assistance agent: {str(e)}", exc_info=True)
            return f"Error providing reasoning assistance: {str(e)}"