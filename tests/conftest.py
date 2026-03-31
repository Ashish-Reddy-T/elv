"""Pytest configuration and shared fixtures.

Registers custom markers:
  - slow: tests that require HuggingFace model downloads or GPU.
    Run with:  pytest tests/ -m slow
    Skip with: pytest tests/ -m "not slow"   (used by `make test-unit`)
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests that require model downloads or GPU (skip with -m 'not slow')",
    )
