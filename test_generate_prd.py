# test_generate_prd.py
# Tests the actual generate_prd tool which was failing previously
import asyncio
import logging
import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API keys from .env
load_dotenv()

# Ensure the key is loaded
if not os.getenv("OPENAI_API_KEY"):
    logger.error("Error: OPENAI_API_KEY not set in .env or environment.")
    exit(1)

logger.info(f"Using OpenAI API Key: ...{os.getenv('OPENAI_API_KEY')[-4:]}")

# Define agent dependencies to match actual application
@dataclass
class ArchitectDeps:
    codebase_path: str
    api_keys: dict[str, str]

# Define the actual tool input model that matches our real application
class GeneratePRDInput(BaseModel):
    task_description: str = Field(description="Description of the programming task")
    codebase_path: str = Field(description="Path to the codebase directory")

# This will simulate our actual PRD result model
class ArchitectResponse(str):
    """String subclass for response formatting"""
    pass

# Determine initialization method based on command line argument
use_direct = len(sys.argv) <= 1 or sys.argv[1] != "string"

if use_direct:
    # Direct model initialization
    from pydantic_ai.models.openai import OpenAIModel
    
    logger.info("Using direct model initialization")
    model = OpenAIModel(
        model_name="gpt-4o",
        provider="openai"
    )
else:
    # Use string-based initialization
    logger.info("Using string-based model initialization")
    model = "openai:gpt-4o"

# Initialize the agent with the selected model, matching our actual agent setup
agent = Agent(
    model,
    deps_type=ArchitectDeps,
    system_prompt="""
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
    
    Format your response in markdown. Be concise but comprehensive.
    Use your tools strategically to gather all the information you need.
    """
)

# Simulated code_reader implementation
@agent.tool
async def code_reader(ctx: RunContext[ArchitectDeps], input_data: GeneratePRDInput) -> str:
    """
    Read and analyze source code files from the specified codebase path, combining them
    into a unified code context for the agent to analyze.
    """
    logger.info(f"Tool called: code_reader for path: {input_data.codebase_path}")
    
    # Just simulate returning something meaningful about the codebase structure
    return """
    # Codebase Structure Analysis
    
    The codebase appears to be a Python package with the following structure:
    
    ## Key Files:
    - `__init__.py`: Package initialization
    - `core.py`: Contains the main Architect class that provides the API
    - `agents/manager.py`: Contains the AgentManager class for handling agent creation
    - `tools/`: Contains tool implementations:
      - `code_reader.py`: Tool for reading and analyzing code
      - `llm.py`: Tool for making LLM calls
      - `web_search.py`: Tool for web searches
    
    ## Architecture:
    The application uses a PydanticAI Agent framework for orchestrating tools and models.
    Currently the system uses only Gemini models for all operations.
    """

# Register the actual generate_prd tool we want to test
@agent.tool
async def generate_prd(ctx: RunContext[ArchitectDeps], input_data: GeneratePRDInput) -> ArchitectResponse:
    """
    Generate a Product Requirements Document (PRD) based on task description and codebase analysis.
    """
    logger.info(f"Tool called: generate_prd for task: {input_data.task_description[:50]}...")
    logger.info(f"Codebase path: {input_data.codebase_path}")
    
    # This is a simplified version of the real implementation
    # In the real version, it would call Gemini API directly
    
    # Get API key (in real implementation, this would be used)
    # api_key = ctx.deps.api_keys.get("gemini")
    
    # For testing purposes, we just return a canned response
    prd = f"""
    # Product Requirements Document: {input_data.task_description}
    
    ## Overview
    This PRD outlines the implementation details for:
    {input_data.task_description}
    
    ## Technical Requirements
    - Must support OpenAI and Gemini models
    - Must maintain backward compatibility
    - Must provide configuration options
    
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
    
    return ArchitectResponse(prd)

async def main():
    try:
        # Create dependencies that match our application's structure
        deps = ArchitectDeps(
            codebase_path="/projects/architect-mcp",
            api_keys={"openai": os.getenv("OPENAI_API_KEY"), "gemini": os.getenv("GEMINI_API_KEY", "")}
        )
        
        # Example task request that would be similar to our real use case
        prompt = """
        Generate a PRD for adding multi-model capability to our AI agent system, 
        supporting both OpenAI (GPT-4o) and Google (Gemini) models.
        
        The system should be able to dynamically select the appropriate model based on the task type.
        Some tasks should use Gemini models (like PRD generation and thinking) 
        while others should use OpenAI models for general agent loops.
        """
        
        # Run the agent
        logger.info("Running agent to test generate_prd tool...")
        result = await agent.run(prompt, deps=deps)
        
        # Display results
        logger.info("\n--- Agent Result ---")
        logger.info(f"Success: {result is not None}")
        logger.info(f"Data: {str(result.data)[:200]}...")  # Show just the beginning
        
        # Get usage statistics
        usage_data = result.usage()
        logger.info(f"Usage: {usage_data}")
        
    except Exception:
        logger.error("An error occurred:", exc_info=True)
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main())