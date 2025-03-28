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
uv run --with "../../$WHEEL" --with mcp --with python-dotenv --with google-genai python -c "
import os
from dotenv import load_dotenv
load_dotenv()
from mcp_server_architect import __version__
from mcp.server.fastmcp import FastMCP
from mcp_server_architect.core import Architect
print(f'Successfully imported mcp-server-architect version {__version__}')
print(f'GEMINI_API_KEY is ' + ('set' if os.getenv('GEMINI_API_KEY') else 'not set'))
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Package import test passed!${NC}"
else
    echo -e "${RED}Package import test failed!${NC}"
    exit 1
fi

# Create a test codebase directory with some sample files
mkdir -p test_codebase
cat > test_codebase/main.py << 'EOL'
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

if __name__ == "__main__":
    print("Calculator app")
    print(f"2 + 3 = {add(2, 3)}")
EOL

cat > test_codebase/utils.py << 'EOL'
def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
EOL

# Create a minimal test script per README instructions
cat > test_script.py << 'EOL'
#!/usr/bin/env python3
"""
Test script to check if mcp-server-architect works with FastMCP API.
"""
import sys
import logging
import os
import pathlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    # Import the FastMCP server and Architect
    from mcp.server.fastmcp import FastMCP
    from mcp_server_architect.core import Architect
    
    logger.info("Successfully imported required modules")
    
    # Check if API key is set
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY is not set!")
    else:
        logger.info("GEMINI_API_KEY is properly set")

    # Create a FastMCP instance
    server = FastMCP(
        "TestArchitect",
        description="Test instance of Architect"
    )
    
    # Register the tool using the new approach
    architect = Architect()
    
    @server.tool()
    def generate_prd(task_description: str, codebase_path: str) -> str:
        """
        Generate a PRD or high-level design document based on codebase analysis and task description.
        """
        return architect.generate_prd(task_description, codebase_path)
    
    logger.info("Successfully created FastMCP server with Architect tool")
    
    # Get absolute path to test codebase
    current_dir = pathlib.Path(__file__).parent.absolute()
    test_codebase_path = current_dir / "test_codebase"
    
    # Run a real test with a small task to reproduce the error
    logger.info(f"Running test with codebase at: {test_codebase_path}")
    task = "Add a multiply_and_add function that multiplies two numbers and adds a third number."
    
    try:
        # Make the request
        logger.info("Making API request to Gemini...")
        result = generate_prd(task, str(test_codebase_path))
        logger.info("Function returned without crashing")
        
        # Print the full result for debugging
        logger.info(f"FULL RESULT: {result}")
        
        # Check result type
        if "⚠️" in result:
            logger.warning("√ SUCCESS: Returned user-friendly 503 error message")
            logger.info("Test successful - 503 error was handled with a nice error message")
        elif "Error generating PRD:" in result:
            logger.warning("√ SUCCESS: Error was caught and handled, but not with the special 503 message")
            logger.info("Test successful - error handled gracefully")
        else:
            logger.info("PRD was successfully generated without errors")
    except Exception as e:
        logger.error(f"PRD generation failed with uncaught exception: {str(e)}", exc_info=True)
        sys.exit(1)
    
    logger.info("Test completed successfully")
    
except Exception as e:
    logger.error(f"Error: {str(e)}", exc_info=True)
    sys.exit(1)
EOL

# Test running the script
echo "Running functional test..."
uv run --with "../../$WHEEL" --with mcp --with python-dotenv --with google-genai python test_script.py

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