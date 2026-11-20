"""Scoring the hand labels against the pipeline.

Two readers label pairs the pipeline decided on; the score has to say how
well they agreed, and then how often the pipeline was right by their
consensus, weighted so that each item stands for the pairs of its cell.
"""
import csv

import pytest

from locus.experiments.annotation_score import (
    DEFINITIONS,
    LABEL_COLUMNS,
    agreement,
    by_stratum,
    effect_on_result,
    estimate,
    is_anchor,
    kappa,
    merge_labels,
    proxy_agreement,
    read_labels,
    score,
    write_labels,
)

# ---- agreement --------------------------------------------------------------


def test_kappa_is_one_on_identical_labels_and_zero_at_chance():
    assert kappa(list("GSWN"), list("GSWN")) == (1.0, 1.0)
    # Two readers who each say G half the time, independently: raw 0.5,
    # expected 0.5, kappa 0.
    raw, k = kappa(list("GGSS"), list("GSGS"))
    assert raw == 0.5
    assert k == pytest.approx(0.0)


def test_kappa_matches_a_worked_example():
    # Landis & Koch style 2x2: 20 agree yes, 5 agree no, 10 and 15 split.
    a = ["y"] * 30 + ["n"] * 20
    b = ["y"] * 20 + ["n"] * 10 + ["y"] * 15 + ["n"] * 5
    raw, k = kappa(a, b)
    assert raw == 0.5
    pe = (30 * 35 + 20 * 15) / 2500
    assert k == pytest.approx((0.5 - pe) / (1 - pe))


def _labels(pairs: dict[str, tuple[str, str]], consensus: dict[str, str] | None = None):
    out = {}
    for item, (a, b) in pairs.items():
        c = (consensus or {}).get(item, a if a == b else None)
        out[item] = {"item_id": item, "label_A": a, "label_B": b,
                     "consensus": c, "adjudicated": a != b}
    return out


def test_agreement_reports_the_confusion_matrix_in_reader_order():
    lab = _labels({"L1": ("G", "G"), "L2": ("G", "S"), "L3": ("W", "W"), "L4": ("N", "W")})
    ag = agreement(lab)
    assert ag["items"] == 4 and ag["disagreements"] == 2
    assert ag["raw"] == 0.5
    assert ag["confusion_A_by_B"]["G"]["S"] == 1
    assert ag["confusion_A_by_B"]["N"]["W"] == 1
    assert ag["confusion_A_by_B"]["S"]["G"] == 0


# ---- consensus --------------------------------------------------------------


def test_merge_takes_the_shared_label_and_the_adjudication_where_they_differ():
    rows = merge_labels({"L1": "G", "L2": "S"}, {"L1": "G", "L2": "W"}, {"L2": "S"})
    assert [(r["consensus"], r["adjudicated"]) for r in rows] == [("G", False), ("S", True)]


def test_merge_refuses_an_unresolved_disagreement_or_a_needless_adjudication():
    with pytest.raises(ValueError, match="no consensus"):
        merge_labels({"L1": "G"}, {"L1": "S"}, {})
    with pytest.raises(ValueError, match="readers agreed"):
        merge_labels({"L1": "G"}, {"L1": "G"}, {"L1": "S"})


def test_label_file_round_trips_and_the_pilot_may_lack_a_consensus(tmp_path):
    rows = merge_labels({"L1": "G", "L2": "S"}, {"L1": "G", "L2": "W"}, {"L2": "S"})
    path = tmp_path / "labels.csv"
    write_labels(path, rows)
    with open(path, newline="") as f:
        assert next(csv.reader(f)) == list(LABEL_COLUMNS)
    got = read_labels(path)
    assert got["L2"]["consensus"] == "S" and got["L2"]["adjudicated"] is True
    pilot = tmp_path / "pilot.csv"
    pilot.write_text("item_id,label_A,label_B,consensus,adjudicated\nP1,G,S,,False\n")
    assert read_labels(pilot)["P1"]["consensus"] is None


# ---- post-stratified estimates ----------------------------------------------


def _row(item, stratum, cell, pop, *, clustering, marker, tier="not_in_window",
         citing="p1", context=None, candidate=None):
    return {
        "item_id": item, "stratum": stratum, "cell": cell, "tier": tier,
        "clustering": clustering, "marker": marker, "cell_population": pop,
        "citing_id": citing, "context_id": context or f"{citing}_{item}",
        "candidate": candidate or f"c_{item}",
    }


def test_routes_read_the_key_flags():
    both = _row("L1", "both_routes", "both_routes/group", 10, clustering=True, marker=True)
    none = _row("L2", "disjoint", "disjoint/all", 10, clustering=False, marker=False)
    assert all(is_anchor(both, r) for r in ("clustering", "marker", "union"))
    assert not any(is_anchor(none, r) for r in ("clustering", "marker", "union"))


def test_precision_and_recall_are_weighted_by_cell_population():
    # Two anchor cells: a big one that is always right and a small one that is
    # never right. One non-anchor cell with one miss in two. Items stand for
    # their cells, so the big cell dominates precision and the miss costs
    # recall in proportion to its cell's size.
    rows = [
        _row("L1", "both_routes", "both_routes/group", 900, clustering=True, marker=True, citing="a"),
        _row("L2", "marker_only", "marker_only/beyond", 100, clustering=False, marker=True, citing="b"),
        _row("L3", "near_miss", "near_miss/all", 200, clustering=False, marker=False, citing="c"),
        _row("L4", "near_miss", "near_miss/all", 200, clustering=False, marker=False, citing="d"),
    ]
    labels = {"L1": "G", "L2": "W", "L3": "G", "L4": "W"}
    est = estimate(rows, labels, "group", "union", n_boot=50)
    assert est["tp"] == 900 and est["fp"] == 100 and est["fn"] == 100
    assert est["precision"] == pytest.approx(0.9)
    assert est["recall"] == pytest.approx(0.9)
    # The clustering route owns only the first cell: perfect precision, and
    # the marker-only cell's W is not its false positive.
    clus = estimate(rows, labels, "group", "clustering", n_boot=50)
    assert clus["precision"] == 1.0 and clus["fp"] == 0


def test_the_sentence_definition_counts_s_as_a_site():
    rows = [_row("L1", "both_routes", "both_routes/sentence", 10, clustering=True, marker=True)]
    assert estimate(rows, {"L1": "S"}, "group", "union", n_boot=5)["precision"] == 0.0
    assert estimate(rows, {"L1": "S"}, "sentence", "union", n_boot=5)["precision"] == 1.0
    assert set(DEFINITIONS) == {"group", "sentence"}


def test_bootstrap_is_deterministic_and_brackets_the_point_estimate():
    rows = [
        _row(f"L{i}", "both_routes", "both_routes/group", 100, clustering=True, marker=True,
             citing=f"p{i}")
        for i in range(10)
    ]
    labels = {f"L{i}": ("G" if i < 7 else "W") for i in range(10)}
    one = estimate(rows, labels, "group", "union", n_boot=200, seed=3)
    two = estimate(rows, labels, "group", "union", n_boot=200, seed=3)
    assert one == two
    lo, hi = one["ci"]["precision"]
    assert lo <= one["precision"] <= hi
    assert lo < hi


# ---- the rest of the report -------------------------------------------------


def test_by_stratum_and_proxy_agreement():
    rows = [
        _row("L1", "both_routes", "both_routes/group", 1, clustering=True, marker=True, tier="group"),
        _row("L2", "both_routes", "both_routes/sentence", 1, clustering=True, marker=True, tier="sentence"),
        _row("L3", "disjoint", "disjoint/all", 1, clustering=False, marker=False),
    ]
    labels = {"L1": "G", "L2": "W", "L3": "N"}
    strata = by_stratum(rows, labels)
    assert strata["disjoint"]["share_N"] == 1.0 and strata["both_routes"]["n"] == 2
    px = proxy_agreement(rows, labels)
    # The not_in_window tier is not a judgement and is left out of the rate.
    assert px["assigned"] == 2 and px["agree"] == 1
    assert px["confusion_tier_by_label"]["not_in_window"]["N"] == 1


def test_score_refuses_a_key_item_without_a_label():
    rows = [_row("L1", "disjoint", "disjoint/all", 1, clustering=False, marker=False)]
    with pytest.raises(ValueError, match="without labels"):
        score(rows, {}, n_boot=2)


class _Assoc:
    """A co-citation stand-in: anchor 'x' pushes candidate 'g' to the top."""

    def aggregate(self, cand, anchors, agg):
        return 3.0 if cand == "g" and "x" in anchors else 0.0


def test_effect_removes_an_annotated_false_positive_and_adds_a_false_negative():
    pool = {"context_id": "ctx", "citing_id": "p", "gold": "g",
            "base": {"g": 0.0, "d": 1.0}, "z": {"g": -1.0, "d": 1.0},
            "anchors": frozenset({"x"})}
    rows = [_row("L1", "both_routes", "both_routes/group", 1, clustering=True, marker=True,
                 context="ctx", candidate="x")]
    # The pipeline's anchor 'x' is judged not co-located: removing it drops
    # the rescored pool back to the base.
    eff = effect_on_result(rows, {"L1": "W"}, [pool], _Assoc(), 1.0, "group")
    fp = eff["touched_by_false_positives"]
    assert fp["pools"] == 1 and fp["anchors_removed"] == 1
    assert fp["base"] == 0.5 and fp["pipeline"] == 1.0 and fp["corrected"] == 0.5
    assert eff["touched_by_false_negatives"]["pools"] == 0
    # The mirror image: no pipeline anchor, and the readers say 'x' belongs.
    bare = {**pool, "anchors": frozenset()}
    rows2 = [_row("L1", "near_miss", "near_miss/all", 1, clustering=False, marker=False,
                  context="ctx", candidate="x")]
    eff2 = effect_on_result(rows2, {"L1": "G"}, [bare], _Assoc(), 1.0, "group")
    fn = eff2["touched_by_false_negatives"]
    assert fn["pools"] == 1 and fn["anchors_added"] == 1
    assert fn["pipeline"] == 0.5 and fn["corrected"] == 1.0
