# tests/equivalence/conftest.py
"""Fixtures for the equivalence gate.

These tests read the 11 GB frozen artifact directory, which is gitignored and
never present in CI. They skip cleanly when it is absent, so `pytest` on a
fresh clone stays green while `pytest -m equivalence` locally is meaningful.
"""
import os
from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "equivalence: recomputes frozen numbers; requires LOCUS_WORK_DIR",
    )


@pytest.fixture(scope="session")
def work_dir() -> Path:
    raw = os.environ.get("LOCUS_WORK_DIR")
    if not raw:
        pytest.skip("LOCUS_WORK_DIR is not set")
    path = Path(raw)
    if not (path / "export" / "locus_rank_test.jsonl").exists():
        pytest.skip(f"{path} does not hold the frozen artefacts")
    return path
