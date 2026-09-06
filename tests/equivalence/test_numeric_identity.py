# tests/equivalence/test_numeric_identity.py
from pathlib import Path

import pytest

# Relative, deliberately. tests/ lives at the repo root, not under src/locus/,
# so an absolute `locus.tests...` import would not exist.
from . import golden

pytestmark = pytest.mark.equivalence

GOLDEN = Path(__file__).with_name("golden_digest.json")


def test_digest_matches_golden(work_dir):
    """Every numeric primitive still produces the byte-exact value it did
    before the restructure. A single mismatch names the key that moved."""
    if not GOLDEN.exists():
        pytest.fail(
            f"{GOLDEN.name} is missing. Generate it once against known-good "
            "code with: python -m tests.equivalence.capture"
        )
    expected = golden.load_golden(GOLDEN)
    actual = golden.digest(work_dir)

    assert set(actual) == set(expected), "digest keys changed"
    drifted = {
        k: (expected[k], actual[k]) for k in expected if expected[k] != actual[k]
    }
    assert not drifted, f"{len(drifted)} value(s) drifted: {drifted}"


def test_rank_floor_is_the_frozen_constant(work_dir):
    """The floor is a published number. If this moves, the paper is wrong."""
    expected = golden.load_golden(GOLDEN)
    assert expected["rank_constant_mrr"] == repr(0.2928968253968254)
    assert expected["swap_constant_acc"] == repr(0.5)
