"""A number in the paper is checked against its report AND its definition.

Matching digits is not enough. A report recomputed on a different subset, with
a different averaging unit or a different anchor route, can land on a number
that looks right, or be swapped in for the right one unnoticed. So every claim
names the report, the key, the value as printed, and the fields that pin down
what the value is a measurement OF.
"""
import json

from locus.core.provenance import check_claim, check_map, lookup


def _write(tmp_path, name, payload):
    (tmp_path / name).write_text(json.dumps(payload))


REPORT = {"gate1_primary": {"delta": 0.118141, "n": 35701, "ci": [0.110621, 0.126008]},
          "papers": 2124, "averaging": "paper", "anchor_route": "union"}
CLAIM = {"id": "headline", "report": "rank_report.json", "key": "gate1_primary.delta",
         "paper": "+0.11814",
         "expect": {"gate1_primary.n": 35701, "papers": 2124, "averaging": "paper"}}


def test_lookup_walks_dotted_keys_and_list_indices():
    assert lookup(REPORT, "gate1_primary.ci.1") == 0.126008
    assert lookup(REPORT, "averaging") == "paper"


def test_a_claim_that_matches_its_report_and_its_definition_passes(tmp_path):
    _write(tmp_path, "rank_report.json", REPORT)
    assert check_claim(CLAIM, tmp_path) == []


def test_the_value_is_compared_at_the_precision_the_paper_prints(tmp_path):
    _write(tmp_path, "rank_report.json", REPORT)
    assert check_claim({**CLAIM, "paper": "+0.118"}, tmp_path) == []
    assert check_claim({**CLAIM, "paper": "+0.11815"}, tmp_path) != []


def test_percentages_and_multiples_are_scaled_before_comparison(tmp_path):
    _write(tmp_path, "r.json", {"share": 0.61166, "ratio": 3.5021, "n": 7})
    pin = {"expect": {"n": 7}}
    assert check_claim({"id": "s", "report": "r.json", "key": "share",
                        "paper": "61.2", "scale": 100, **pin}, tmp_path) == []
    assert check_claim({"id": "x", "report": "r.json", "key": "ratio",
                        "paper": "3.50", **pin}, tmp_path) == []


def test_a_matching_number_on_a_different_subset_fails(tmp_path):
    # The digits agree; the report was computed on other pools. This is the
    # failure a digits-only check cannot see.
    _write(tmp_path, "rank_report.json",
           {**REPORT, "gate1_primary": {**REPORT["gate1_primary"], "n": 34270}})
    problems = check_claim(CLAIM, tmp_path)
    assert len(problems) == 1 and "gate1_primary.n" in problems[0]


def test_a_matching_number_under_another_averaging_unit_fails(tmp_path):
    _write(tmp_path, "rank_report.json", {**REPORT, "averaging": "pool"})
    assert any("averaging" in p for p in check_claim(CLAIM, tmp_path))


def test_a_missing_report_or_key_is_a_problem_not_a_crash(tmp_path):
    assert "missing" in check_claim(CLAIM, tmp_path)[0]
    _write(tmp_path, "rank_report.json", {"papers": 2124})
    assert any("gate1_primary.delta" in p for p in check_claim(CLAIM, tmp_path))


def test_a_claim_without_expected_fields_is_rejected(tmp_path):
    # An entry that pins nothing down is the digits-only check again.
    _write(tmp_path, "rank_report.json", REPORT)
    bare = {k: v for k, v in CLAIM.items() if k != "expect"}
    assert any("expect" in p for p in check_claim(bare, tmp_path))


def test_check_map_also_requires_each_printed_value_to_appear_in_the_paper(tmp_path):
    _write(tmp_path, "rank_report.json", REPORT)
    claims = {"claims": [CLAIM]}
    assert check_map(claims, tmp_path, tex="the gain is $+0.11814$ MRR") == []
    problems = check_map(claims, tmp_path, tex="the gain is $+0.12926$ MRR")
    assert len(problems) == 1 and "not found in the paper" in problems[0]


def test_thousands_separators_in_the_paper_are_understood(tmp_path):
    _write(tmp_path, "r.json", {"n": 35701, "averaging": "paper"})
    claim = {"id": "n", "report": "r.json", "key": "n", "paper": "35{,}701",
             "expect": {"averaging": "paper"}}
    assert check_map({"claims": [claim]}, tmp_path, tex="on 35{,}701 pools") == []


def test_lookup_reaches_keys_that_contain_a_dot():
    # Thresholds are report keys ("0.3"), and a dotted path would otherwise
    # split them in two.
    report = {"thetas": {"0.3": {"full_rebuild": {"delta": 0.1192}}}}
    assert lookup(report, "thetas.0.3.full_rebuild.delta") == 0.1192
