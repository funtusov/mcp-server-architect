#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Load environment variables from .env
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    source .env
else
    echo "${RED}Warning: .env file not found. Tests might fail if GEMINI_API_KEY is required.${NC}"
fi

# Clean up previous builds
echo "Cleaning up previous builds..."
rm -rf dist/
rm -rf tmp/test_build/

# Create a temp directory for testing
mkdir -p tmp/test_build

# Update version.py
echo "Checking version numbers..."
VERSION=$(grep "__version__" mcp_server_architect/version.py | cut -d'"' -f2)
echo "Current version: $VERSION"

# Build the package using 'uv build --no-sources'
echo "Building package..."
uv build --no-sources

# Find the latest wheel
WHEEL=$(ls dist/*.whl | sort -V | tail -n1)
echo "Using wheel: $WHEEL"

# Create test directory
echo "Creating test environment..."
cd tmp/test_build

# Create .env file in test directory
echo "Creating .env file in test directory..."
echo "GEMINI_API_KEY=$GEMINI_API_KEY" > .env
echo "GEMINI_MODEL=gemini-2.5-pro-exp-03-25" >> .env

# Test installing and importing from the built package
echo "Testing package import..."
uv run --with "../../$WHEEL" --with mcp --with python-dotenv --with google-generativeai python -c "
import os
from dotenv import load_dotenv
load_dotenv()
from mcp_server_architect import __version__
from mcp.server.fastmcp import FastMCP
from mcp_server_architect.core import ArchitectAI
print(f'Successfully imported mcp-server-architect version {__version__}')
print(f'GEMINI_API_KEY is ' + ('set' if os.getenv('GEMINI_API_KEY') else 'not set'))
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Package import test passed!${NC}"
else
    echo -e "${RED}Package import test failed!${NC}"
    exit 1
fi

# Create a minimal test script per README instructions
cat > test_script.py << 'EOL'
#!/usr/bin/env python3
"""
Test script to check if mcp-server-architect works with FastMCP API.
"""
import sys
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    # Import the FastMCP server and ArchitectAI
    from mcp.server.fastmcp import FastMCP
    from mcp_server_architect.core import ArchitectAI
    
    logger.info("Successfully imported required modules")
    
    # Check if API key is set
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY is not set!")
    else:
        logger.info("GEMINI_API_KEY is properly set")

    # Create a FastMCP instance following the pattern in the fixed code
    server = FastMCP(
        "TestArchitectAI",
        description="Test instance of ArchitectAI"
    )
    
    # Register the tool using the new approach
    architect = ArchitectAI()
    
    @server.tool()
    def generate_prd(task_description: str, codebase_path: str) -> str:
        """
        Generate a PRD or high-level design document based on codebase analysis and task description.
        """
        return architect.generate_prd(task_description, codebase_path)
    
    logger.info("Successfully created FastMCP server with ArchitectAI tool")
    
    # Test a simple call to ensure API connection works
    # Don't need to run the full API call, just verify initialization works
    logger.info("Test completed successfully")
    
except Exception as e:
    logger.error(f"Error: {str(e)}", exc_info=True)
    sys.exit(1)
EOL

# Test running the script
echo "Running functional test..."
uv run --with "../../$WHEEL" --with mcp --with python-dotenv --with google-generativeai python test_script.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Functional test passed!${NC}"
else
    echo -e "${RED}Functional test failed!${NC}"
    exit 1
fi

# Cleanup
rm .env

cd ../..

echo -e "${GREEN}All tests completed successfully!${NC}"
echo "Package is ready for use or publishing!"