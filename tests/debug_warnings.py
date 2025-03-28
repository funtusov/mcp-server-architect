#!/usr/bin/env python3
"""
Debug script to find the source of Pydantic warnings.
"""

import os
import warnings
import traceback

# Custom warning handler to print traceback
def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    print("=" * 80)
    print(f"WARNING: {message}")
    print(f"Category: {category}")
    print(f"File: {filename}:{lineno}")
    # Print stack trace
    print("\nStack trace:")
    traceback.print_stack()
    print("=" * 80)

# Replace the default warning handler
warnings.showwarning = custom_showwarning

# Now import libraries that might cause warnings
print("Importing pydantic...")
import pydantic

print("Importing pydantic_ai...")
import pydantic_ai

print("Importing specific modules from pydantic_ai...")
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

print("Creating a test agent...")
model = TestModel()
agent = Agent(model)

print("Test complete")