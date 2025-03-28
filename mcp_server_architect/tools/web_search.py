#!/usr/bin/env python3
"""
Web search tool for the Architect agent using Exa API.
"""

import logging

from exa_py import Exa
from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from mcp_server_architect.types import ArchitectDependencies

# Configure logging
logger = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    """Input schema for the web search tool."""
    query: str = Field(..., description="The search query to execute")
    num_results: int | None = Field(10, description="Number of results to return (max 25)")


async def web_search(ctx: RunContext[ArchitectDependencies], input_data: WebSearchInput) -> str:
    """
    Search the web for information on a given query using Exa API.
    
    Args:
        ctx: The runtime context containing dependencies
        input_data: The input parameters specifying the search query
        
    Returns:
        A string containing the search results or error message
    """
    try:
        # Get API key from context
        api_key = ctx.deps.api_keys.get("web_search")
        if not api_key:
            return "Error: Web search API key not configured. Please set the WEB_SEARCH_API_KEY environment variable."
            
        # Limit number of results
        num_results = min(input_data.num_results, 25)  # Cap at 25 results
        
        logger.info(f"Performing Exa web search for query: {input_data.query}")
        
        try:
            # Initialize the Exa client
            exa = Exa(api_key=api_key)
            
            # Perform search with full text contents
            search_results = exa.search_and_contents(
                input_data.query,
                num_results=num_results,
                use_autoprompt=True,  # Enable autoprompt for better results
                text=True  # Get full text of results
            )
            
            # Format and return results
            return _format_exa_search_results(search_results)
            
        except Exception as e:
            error_message = f"Exa API request failed: {str(e)}"
            logger.error(error_message)
            return error_message
            
    except Exception as e:
        logger.error(f"Unexpected error in web_search tool: {str(e)}", exc_info=True)
        return f"Error in web search tool: {str(e)}"


def _format_exa_search_results(search_results) -> str:
    """Format Exa search results into a readable string."""
    try:
        results = search_results.results
        if not results:
            return "No search results found."
            
        formatted_results = ["# Search Results\n"]
        
        for i, result in enumerate(results, 1):
            title = result.title if hasattr(result, 'title') and result.title else "No title"
            url = result.url if hasattr(result, 'url') and result.url else "No URL"
            
            formatted_results.append(f"## {i}. {title}\n")
            formatted_results.append(f"URL: {url}\n")
            
            # Add published date if available
            if hasattr(result, 'published_date') and result.published_date:
                formatted_results.append(f"Published: {result.published_date}\n")
            
            # Add full text content if available
            if hasattr(result, 'text') and result.text:
                text = result.text
                # Trim long content to a reasonable size to avoid overwhelming the LLM
                if len(text) > 3000:
                    formatted_results.append(f"\nContent (truncated):\n{text[:3000]}...\n")
                else:
                    formatted_results.append(f"\nContent:\n{text}\n")
            
            # Add separator between results
            formatted_results.append("\n---\n")
            
        # Return formatted results
        return "\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Error formatting search results: {str(e)}", exc_info=True)
        return f"Error formatting search results: {str(e)}"