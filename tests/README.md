# Architect MCP Integration Tests

This directory contains integration tests for the Architect MCP Server.

## Running Tests

To run the tests:

```bash
# Install development dependencies
uv add -e .[dev]

# Run all tests
pytest

# Run specific test file
pytest tests/test_integration.py

# Run with verbose output
pytest -v
```

## Recording Mode

The tests use pytest-recording to record and replay HTTP interactions. When running tests for the first time or when you need to update the recordings:

```bash
# Force re-recording
pytest --record-mode=rewrite

# For a specific test
pytest tests/test_integration.py::test_generate_prd --record-mode=rewrite
```

## Test Environment

Tests require the following environment variables:

- `GEMINI_API_KEY` - API key for Google Gemini
- `EXA_API_KEY` - API key for Exa search

During tests, these are automatically replaced with mock values, but real API keys are needed when recording new cassettes.

## VCR Configuration

VCR is configured to:

1. Filter out sensitive information like API keys
2. Store cassettes in the `tests/cassettes` directory
3. Use the `once` record mode by default (record once then replay)

See `conftest.py` for detailed VCR configuration.