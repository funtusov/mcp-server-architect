# test_full_simulation.py
# A more complex test that closely mimics the real application structure
import asyncio
import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load API keys from .env
load_dotenv()

# Ensure OpenAI key is loaded
if not os.getenv("OPENAI_API_KEY"):
    logger.error("Error: OPENAI_API_KEY not set in .env or environment.")
    exit(1)

logger.info(f"Using OpenAI API Key: ...{os.getenv('OPENAI_API_KEY')[-4:]}")


# Define more complex agent dependencies that match our real app
@dataclass
class ArchitectDependencies:
    """Dependencies for the Architect agent, matching the real application."""

    codebase_path: str
    api_keys: dict[str, str]

    # Add a helper method to get API keys, similar to real implementation
    def get_api_key(self, provider: str) -> str:
        """Get API key for a specific provider."""
        return self.api_keys.get(provider, "")


# Input models for the various tools
class CodeReaderInput(BaseModel):
    """Input model for the code_reader tool."""

    codebase_path: str = Field(description="Path to the codebase directory")
    file_patterns: list[str] = Field(description="List of file patterns to include", default=["**/*.py"])


class WebSearchInput(BaseModel):
    """Input model for the web_search tool."""

    query: str = Field(description="The search query")
    num_results: int = Field(description="Number of results to return", default=3)


class GeneratePRDInput(BaseModel):
    """Input model for the generate_prd tool."""

    task_description: str = Field(description="Description of the programming task")
    codebase_path: str = Field(description="Path to the codebase directory")


class LLMInput(BaseModel):
    """Input model for the llm tool."""

    prompt: str = Field(description="The prompt to send to the LLM")
    model: str = Field(description="The model to use", default="")


# In the actual problem case, we're getting a 404 "tools is not supported" error
# Only for openai:gpt-4o string-based initialization


# Add result type to make it more similar to the problematic case
class ArchitectResult(BaseModel):
    """The result type for the agent."""

    content: str = Field(description="The generated PRD content")


# Create the agent using string initialization
agent = Agent(
    "openai:gpt-4o",
    deps_type=ArchitectDependencies,
    result_type=ArchitectResult,  # Adding result_type might be contributing to the issue
    system_prompt="""
    You are an expert software architect and technical lead.
    
    Your task is to create a Product Requirements Document (PRD) or High-Level Design Document based on the 
    user's request and any code context you gather using your tools.
    
    You have access to several tools:
    - code_reader: Read and analyze source code from the codebase
    - web_search: Search the web for relevant information
    - llm: Call an LLM for specialized analysis
    
    Your PRD should include:
    1. Overview of the requested feature/task
    2. Technical requirements and constraints
    3. Proposed architecture/design
    4. Implementation plan with specific files to modify
    5. Potential challenges and mitigations
    
    Format your response in markdown. Be concise but comprehensive.
    """,
)


# Implement code_reader tool
@agent.tool
async def code_reader(ctx: RunContext[ArchitectDependencies], input_data: CodeReaderInput) -> str:
    """
    Read and analyze source code files from the specified codebase path, combining them
    into a unified code context for the agent to analyze.
    """
    logger.info(f"Tool called: code_reader for path: {input_data.codebase_path}")
    logger.info(f"File patterns: {input_data.file_patterns}")

    # Simulate file reading and add complexity with error handling
    try:
        # Simulate some async work
        await asyncio.sleep(0.5)

        # In the real implementation, this would actually read files
        return """
        # Codebase Structure Analysis
        
        The codebase is a Python package with the following structure:
        
        ## Key Files and Modules:
        - `__init__.py`: Package initialization with version
        - `core.py`: Contains the main Architect class that provides the API
        - `agents/manager.py`: Contains the AgentManager class for handling agent creation
        - `models.py`: Contains model configuration for different LLMs
        - `tools/`: Contains tool implementations:
          - `code_reader.py`: Tool for reading and analyzing code
          - `llm.py`: Tool for making LLM calls
          - `web_search.py`: Tool for web searches
        
        ## Configuration:
        The application uses environment variables:
        - GEMINI_API_KEY: For Google Gemini models
        - OPENAI_API_KEY: For OpenAI models
        - EXA_API_KEY: For web search functionality
        
        ## Architecture:
        The application uses a PydanticAI Agent framework for orchestrating tools and models.
        Currently the system uses only Gemini models for all operations.
        """
    except Exception as e:
        logger.error(f"Error in code_reader tool: {str(e)}")
        return f"Error analyzing codebase: {str(e)}"


# Implement web_search tool
@agent.tool
async def web_search(ctx: RunContext[ArchitectDependencies], input_data: WebSearchInput) -> str:
    """
    Search the web for information related to the query.
    """
    logger.info(f"Tool called: web_search for query: {input_data.query}")

    # Check if we have an API key for web search
    api_key = ctx.deps.get_api_key("web_search")
    if not api_key:
        logger.warning("No web search API key found, returning mock results")

    # In a real implementation, this would make an API call
    try:
        # Simulate latency
        await asyncio.sleep(1)

        # Use the LLM tool within this tool - this nested tool pattern might be causing issues
        llm_result = await llm(
            ctx,
            LLMInput(
                prompt=f"Generate search results for: {input_data.query}",
                model="",  # Use default model
            ),
        )

        logger.info("Generated search results with llm tool")

        # Return simulated search results
        return f"""
        # Search Results for: {input_data.query}
        
        ## Result 1
        PydanticAI documentation for Agent class
        The Agent class provides tools for running LLM-powered agents with function calling.
        
        ## Result 2
        Multi-model capabilities in AI systems
        Strategies for using different AI models for different tasks based on capabilities.
        
        ## Result 3
        {llm_result}
        """
    except Exception as e:
        logger.error(f"Error in web_search tool: {str(e)}")
        return f"Error performing web search: {str(e)}"


# Implement llm tool - this is more complex in the real app
@agent.tool
async def llm(ctx: RunContext[ArchitectDependencies], input_data: LLMInput) -> str:
    """
    Execute a prompt against a configured language model.
    """
    logger.info(f"Tool called: llm with prompt: {input_data.prompt[:50]}...")

    # In real implementation, this would use a different model (likely Gemini)
    # and make direct API calls instead of using PydanticAI
    try:
        # Simulate API call latency
        await asyncio.sleep(0.5)

        # Return simulated LLM response
        return f"""
        Analysis for: {input_data.prompt[:30]}...
        
        Based on my analysis, here are some key considerations:
        1. Implementation complexity varies based on chosen model providers
        2. OpenAI models excel at reasoning tasks
        3. Google Gemini models have strong code understanding
        4. A model registry pattern is recommended for multi-model implementations
        """
    except Exception as e:
        logger.error(f"Error in llm tool: {str(e)}")
        return f"Error processing LLM prompt: {str(e)}"


# Implement generate_prd tool
@agent.tool
async def generate_prd(ctx: RunContext[ArchitectDependencies], input_data: GeneratePRDInput) -> str:
    """
    Generate a Product Requirements Document (PRD) based on task description and codebase analysis.
    """
    logger.info(f"Tool called: generate_prd for task: {input_data.task_description[:50]}...")

    # In the real implementation, this would likely call the Gemini API directly
    try:
        # Check if we have a Gemini API key
        api_key = ctx.deps.get_api_key("gemini")
        if not api_key:
            logger.warning("No Gemini API key found, using simulated response")

        # Simulate API call to Gemini
        await asyncio.sleep(1)

        # Use the LLM tool internally to simulate the real implementation
        # This nested tool pattern might be causing issues
        analysis = await llm(
            ctx,
            LLMInput(
                prompt=f"Analyze the implementation needs for: {input_data.task_description}",
                model="gemini-2.5",  # This might be ignored in our simulation
            ),
        )

        # Return simulated PRD with the analysis
        return f"""
        # Product Requirements Document
        
        ## Overview
        This PRD outlines the implementation details for:
        {input_data.task_description}
        
        ## Technical Analysis
        {analysis}
        
        ## Proposed Architecture
        - Create a models.py file with model configuration
        - Update AgentManager to support different model types
        - Add environment variable control
        
        ## Implementation Plan
        1. Create models.py with model registry
        2. Update AgentManager._create_agent method
        3. Add model selection logic to tools
        4. Update tests
        
        ## Challenges and Mitigations
        - API compatibility issues: Use adapter pattern
        - Testing complexity: Create test fixtures for each model
        """
    except Exception as e:
        logger.error(f"Error in generate_prd tool: {str(e)}", exc_info=True)
        return f"Error generating PRD: {str(e)}"


async def main():
    try:
        # Set up dependencies similar to the real application
        deps = ArchitectDependencies(
            codebase_path="/projects/architect-mcp",
            api_keys={
                "openai": os.getenv("OPENAI_API_KEY"),
                "gemini": os.getenv("GEMINI_API_KEY", ""),
                "web_search": os.getenv("EXA_API_KEY", ""),
            },
        )

        # Example task request that would trigger multiple tools - make it more complex
        prompt = """
        Generate a comprehensive PRD for adding multi-model capability to our AI agent system, 
        supporting both OpenAI (GPT-4o) and Google (Gemini) models.
        
        The system should be able to dynamically select the appropriate model based on the task type.
        Some tasks should use Gemini models (like PRD generation and thinking) 
        while others should use OpenAI models for general agent loops.
        
        Your PRD should:
        1. Research best practices for multi-model architectures
        2. Consider API compatibility between different providers
        3. Design a model registry system with environment variable configuration
        4. Outline error handling for API failures
        5. Consider performance implications of different models
        6. Detail the implementation plan with specific files to modify
        
        Make sure to analyze the existing codebase structure for compatibility.
        """

        # Add model settings - this is the correct parameter name in this API version
        model_settings = {
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 2000,
        }

        # Run the agent - using the same pattern as in the real app
        logger.info("Running agent with complex tool set...")
        try:
            # Add more parameters to make it closer to the problem scenario
            # Use model parameter explicitly - this might be the key difference
            result = await agent.run(
                prompt,
                deps=deps,
                model_settings=model_settings,
                model="openai:gpt-4o",  # Explicitly set model again, which might create conflict
                infer_name=True,  # Add additional parameters from signature
                result_type=ArchitectResult,  # Set result type again
            )

            # Display results
            logger.info("\n--- Agent Result ---")
            logger.info(f"Success: {result is not None}")
            first_200 = (
                str(result.data.content)[:200] + "..."
                if len(str(result.data.content)) > 200
                else str(result.data.content)
            )
            logger.info(f"Data: {first_200}")

            # Get usage statistics
            usage_data = result.usage()
            logger.info(f"Usage: {usage_data}")
        except Exception as e:
            # This is to catch the specific error we're looking for
            logger.error(f"Agent run failed: {str(e)}")
            if "tools is not supported" in str(e):
                logger.error("FOUND THE ERROR! This is the same error as in production.")
            if "404" in str(e):
                logger.error("FOUND A 404 ERROR! This might be related to our production issue.")

    except Exception:
        logger.error("An error occurred:", exc_info=True)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
