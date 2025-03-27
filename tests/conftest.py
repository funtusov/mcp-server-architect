#!/usr/bin/env python3

import os
import pytest
from unittest.mock import patch

# Patch environmental variables to avoid requiring actual API keys during tests
@pytest.fixture(autouse=True)
def mock_env_variables():
    with patch.dict(os.environ, {
        "GOOGLE_API_KEY": "dummy_api_key",
        "GEMINI_MODEL": "gemini-2.5-pro-exp-03-25"
    }):
        yield

# Patch Google Generative AI configuration
@pytest.fixture(autouse=True)
def mock_genai_configure():
    with patch('google.generativeai.configure'):
        yield