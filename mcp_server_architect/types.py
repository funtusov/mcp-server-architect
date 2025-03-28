#!/usr/bin/env python3
"""
Shared types and models for Architect.
"""

from dataclasses import dataclass


@dataclass
class ArchitectDependencies:
    """Dependencies for the ArchitectAgent."""
    codebase_path: str
    api_keys: dict[str, str]