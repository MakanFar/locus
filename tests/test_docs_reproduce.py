"""The numbers docs/reproduce.md tells a reviewer to expect must be real.

A reproduction guide that names a wrong number is worse than one that names
none: the reviewer concludes the artefact is broken. The floors and counts
are read out of the exported manifest rather than restated.
"""
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "reproduce.md"

pytestmark = pytest.mark.equivalence


@pytest.fixture
def manifest():
    work = os.environ.get("LOCUS_WORK_DIR")
    if not work:
        pytest.skip("needs LOCUS_WORK_DIR")
    path = Path(work) / "export" / "locus_manifest_test.json"
    if not path.exists():
        pytest.skip(f"no manifest at {path}")
    return json.loads(path.read_text())


def test_every_floor_the_guide_quotes_matches_the_manifest(manifest):
    doc = DOC.read_text()
    for key, value in manifest["floors"].items():
        if key == "pool_size":
            continue
        assert str(value) in doc, f"reproduce.md does not quote {key} = {value}"


def test_every_count_the_guide_quotes_matches_the_manifest(manifest):
    doc = DOC.read_text()
    # Written with thousands separators in prose, without them in JSON.
    for key in ("rank_items", "headline", "contexts", "citing_papers"):
        n = manifest["counts"][key]
        assert f"{n:,}" in doc or str(n) in doc, f"reproduce.md omits {key} = {n}"


def test_the_headline_result_is_quoted_exactly():
    doc = DOC.read_text()
    for number in ("0.47771", "0.59585", "+0.11814"):
        assert number in doc


def test_the_guide_does_not_promise_a_deposit_that_does_not_exist():
    # If locus fetch has no DEFAULT_MANIFEST_URL, the guide must say
    # --manifest-url is required rather than print a command that cannot work.
    from locus.data import fetch

    doc = DOC.read_text()
    if fetch.DEFAULT_MANIFEST_URL is None:
        assert "--manifest-url" in doc
