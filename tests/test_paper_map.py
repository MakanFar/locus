"""reports/paper_map.json must hold against the reports committed beside it.

The map ties each number the paper prints to a report, a key and the fields
that define what was measured. This test is the half that needs no paper
source: every claim's value and every expected field is checked against the
committed report. `python -m locus.core.provenance --map reports/paper_map.json
--tex <main.tex>` adds the other half, that each printed value occurs in the
paper.
"""
import json
from pathlib import Path

from locus.core.provenance import check_map

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def test_every_claim_in_the_map_holds_against_its_report():
    claims = json.loads((REPORTS / "paper_map.json").read_text())
    assert claims["claims"], "the map lists no claims"
    assert check_map(claims, REPORTS) == []


def test_claim_ids_are_unique():
    claims = json.loads((REPORTS / "paper_map.json").read_text())["claims"]
    ids = [c["id"] for c in claims]
    assert len(ids) == len(set(ids))
