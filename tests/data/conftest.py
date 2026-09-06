"""Fixtures shared across tests/data/.

`second_corpus` used to live here; it moved to tests/conftest.py when the
custom-corpus path grew tests under tests/build/ as well.
"""
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
