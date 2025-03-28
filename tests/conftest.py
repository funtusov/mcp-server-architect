#!/usr/bin/env python3
"""
Configuration for pytest.
"""

import os

import pytest
from dotenv import load_dotenv
from vcr.filters import decode_response

# Load environment variables from .env file
load_dotenv()

# Create directory for cassettes if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), "cassettes"), exist_ok=True)

# Check if API keys are available
if not os.environ.get("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY environment variable is not set. Tests require a valid API key.")

# Optional EXA API key check
if not os.environ.get("EXA_API_KEY"):
    print("WARNING: EXA_API_KEY environment variable is not set. Web search tools will not work in tests.")


def filter_api_keys(response):
    """Filter out API keys from responses to avoid storing them in cassettes."""
    body = None
    
    # Handle different response formats
    if isinstance(response, dict) and "body" in response:
        body = response["body"]
    
    # Filter sensitive data if body exists
    if body:
        # Look for common API key patterns
        for key in ["access_token", "api_key", "token", "key"]:
            if key in body:
                body[key] = "[FILTERED]"  # noqa: S105
    
    return response


# Configure VCR for pytest-recording
pytest.recording_vcr_kwargs = {
    "filter_headers": [
        ("authorization", "[FILTERED]"),
        ("x-goog-api-key", "[FILTERED]"),
    ],
    "decode_compressed_response": True,
    "filter_query_parameters": [
        ("key", "[FILTERED]"),
        ("api_key", "[FILTERED]"),
    ],
    "before_record_response": [
        decode_response,
        filter_api_keys
    ],
    "record_mode": "once",
    "path_transformer": lambda path: os.path.join(
        os.path.dirname(__file__), "cassettes", os.path.basename(path)
    ),
    "match_on": ["method", "scheme", "host", "port", "path", "query"],
}