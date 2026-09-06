import pathlib

import pytest

from locus import config
from locus.core.types import ContextRec
from locus.eval import rank as rank_mod
from locus.eval.rank import other_location_anchors, paired


def _rec(cid, refid, loc):
    return ContextRec(
        context_id=f"{cid}_{refid}_{loc}", citing_id=cid, refid=refid,
        raw="", masked="w", location=loc,
    )


INDEX = {
    "p1": [_rec("p1", "a", 0), _rec("p1", "b", 0),
           _rec("p1", "c", 1), _rec("p1", "d", 2)],
    "p2": [_rec("p2", "e", 0), _rec("p2", "f", 0)],   # one location only
}


def _patch(monkeypatch, index=INDEX):
    monkeypatch.setattr(rank_mod, "load_index", lambda path: index)
    monkeypatch.setattr(config, "WORK_DIR", pathlib.Path("/nonexistent"))


def test_control_anchors_come_from_a_different_location(monkeypatch):
    _patch(monkeypatch)
    got = other_location_anchors("test", seed=0)
    # p1 location 0 golds {a,b}, location 1 {c}, location 2 {d}
    assert got["p1_a_0"] in (frozenset({"c"}), frozenset({"d"}))
    assert got["p1_c_1"] in (frozenset({"a", "b"}), frozenset({"d"}))
    for cid, own in (("p1_a_0", {"a", "b"}), ("p1_c_1", {"c"}), ("p1_d_2", {"d"})):
        assert got[cid] != frozenset(own)


def test_single_location_paper_gets_no_control(monkeypatch):
    # Handing these an empty anchor set would compare "other-location anchors"
    # against "no anchors" and silently answer a different question.
    _patch(monkeypatch)
    got = other_location_anchors("test", seed=0)
    assert "p2_e_0" not in got
    assert "p2_f_0" not in got


def test_control_choice_is_deterministic(monkeypatch):
    # Seeded per citing paper and drawn from a sorted list, so it must not
    # depend on dict iteration order or on PYTHONHASHSEED.
    _patch(monkeypatch)
    assert other_location_anchors("test", 0) == other_location_anchors("test", 0)
    assert other_location_anchors("test", 1) != {} 


def test_paired_delta_is_the_mean_of_differences(monkeypatch):
    a = [0.9, 0.5, 0.4]
    b = [0.4, 0.5, 0.1]
    groups = ["p1", "p1", "p2"]
    got = paired(a, b, groups, seed=0)
    # paper p1 mean diff = (0.5 + 0.0)/2 = 0.25; p2 = 0.3; mean = 0.275
    assert got["delta"] == pytest.approx(0.275)
    assert got["n"] == 3
    assert got["ci"][0] <= got["delta"] <= got["ci"][1]


def test_paired_reports_a_zero_straddling_interval_as_such():
    a = [0.5, 0.1, 0.9, 0.2]
    b = [0.1, 0.5, 0.2, 0.9]   # differences cancel across papers
    got = paired(a, b, ["p1", "p1", "p2", "p2"], seed=0)
    assert got["delta"] == pytest.approx(0.0)
    assert not got["excludes_zero"]


def test_pairing_shows_up_in_the_interval_not_the_point_estimate():
    # paper_mean(a) - paper_mean(b) == paper_mean(a - b) algebraically, so the
    # delta alone cannot tell a paired analysis from an unpaired one. The
    # difference is entirely in the interval: here every item improves by
    # exactly 0.1 while the underlying scores swing across the whole range, so
    # the paired interval is zero-width and an unpaired one would be wide.
    b = [0.05, 0.85, 0.10, 0.90, 0.45]
    a = [x + 0.1 for x in b]
    groups = ["p1", "p1", "p2", "p2", "p3"]
    got = paired(a, b, groups, seed=0)
    assert got["delta"] == pytest.approx(0.1)
    lo, hi = got["ci"]
    assert hi - lo == pytest.approx(0.0, abs=1e-12)
    assert got["excludes_zero"]


def _pool(scores, gold="g"):
    return {"base": scores, "z": scores, "gold": gold,
            "anchors": frozenset(), "citing_id": "p", "context_id": "c"}


class _ZeroPPMI:
    def aggregate(self, c, anchors, how="mean"):
        return 0.0


class _RecordingPPMI:
    """Returns a fixed value and remembers which aggregator it was asked for."""

    def __init__(self, value=1.0):
        self.value = value
        self.seen = []

    def aggregate(self, c, anchors, how="mean"):
        self.seen.append(how)
        return self.value


def test_score_forwards_the_aggregator_on_every_metric():
    # The R@1 gate calls _score with the metric POSITIONALLY, so an aggregator
    # threaded only through the MRR call sites would leave the R@1 column
    # silently computed under the mean while the table claimed otherwise --
    # a wrong number with nothing to flag it.
    pool = _pool({"g": 1.0, "x": 2.0})
    for metric in (rank_mod.reciprocal_rank, rank_mod.hit_at_1):
        ppmi = _RecordingPPMI()
        rank_mod._score(pool, ppmi, frozenset(["a"]), 3.0, metric, "masked_mean")
        assert set(ppmi.seen) == {"masked_mean"}, metric


def test_score_honours_the_requested_metric():
    # gold second: MRR 0.5, hit@1 0.0. A _score that ignored `metric` and
    # always returned the reciprocal rank would pass the first assert and
    # fail the second.
    pool = _pool({"g": 1.0, "x": 2.0})
    assert rank_mod._score(pool, _ZeroPPMI(), frozenset(), 0.0,
                       rank_mod.reciprocal_rank) == pytest.approx(0.5)
    assert rank_mod._score(pool, _ZeroPPMI(), frozenset(), 0.0,
                       rank_mod.hit_at_1) == pytest.approx(0.0)


def test_score_metric_applies_on_the_rescored_path_too():
    # lam != 0 takes the other branch of _score. Both branches must respect
    # the metric; the lam == 0 shortcut reads pool["base"] and the rescored
    # path reads pool["z"], so they are genuinely separate code.
    pool = _pool({"g": 1.0, "x": 2.0})
    assert rank_mod._score(pool, _ZeroPPMI(), frozenset(), 3.0,
                       rank_mod.hit_at_1) == pytest.approx(0.0)
    top = _pool({"g": 5.0, "x": 2.0})
    assert rank_mod._score(top, _ZeroPPMI(), frozenset(), 3.0,
                       rank_mod.hit_at_1) == pytest.approx(1.0)


def test_hit_at_1_and_mrr_disagree_on_a_tied_pool():
    # A 2-way tie: hit@1 is 1/2 and MRR is (1/1 + 1/2)/2 = 3/4. Reporting one
    # under the other convention would put 0.75 in the R@1 column.
    pool = _pool({"g": 1.0, "x": 1.0})
    assert rank_mod._score(pool, _ZeroPPMI(), frozenset(), 0.0,
                       rank_mod.hit_at_1) == pytest.approx(0.5)
    assert rank_mod._score(pool, _ZeroPPMI(), frozenset(), 0.0,
                       rank_mod.reciprocal_rank) == pytest.approx(0.75)


class _TablePPMI:
    """PPMI from a (candidate, anchor) -> value table; mean over the anchor set."""

    def __init__(self, table):
        self.table = table

    def aggregate(self, c, anchors, how="mean"):
        if not anchors:
            return 0.0
        return sum(self.table.get((c, a), 0.0) for a in anchors) / len(anchors)


def _pool_with(scores, anchors, gold="g", citing="p", ctx="c"):
    return {"base": scores, "z": scores, "gold": gold,
            "anchors": frozenset(anchors), "citing_id": citing, "context_id": ctx}


class TestCoverageStratum:
    """Table 6: what the training graph knows about each headline pool.

    A pool is `target` when PPMI(gold, A) > 0, `distractor` when the gold is
    unseen but some distractor is co-cited with the anchors, and `none` when
    the graph is silent on the whole pool. The three are exhaustive and
    exclusive, and `target` wins whenever it applies -- a distractor also being
    seen does not demote a pool whose gold relation the graph captured.
    """

    def test_target_wins_even_when_a_distractor_is_also_seen(self):
        from locus.eval.rank import coverage_stratum
        ppmi = _TablePPMI({("g", "a"): 5.0, ("x", "a"): 9.0})
        pool = _pool_with({"g": 1.0, "x": 2.0}, ["a"])
        assert coverage_stratum(pool, ppmi) == "target"

    def test_distractor_when_only_a_distractor_is_co_cited(self):
        from locus.eval.rank import coverage_stratum
        ppmi = _TablePPMI({("x", "a"): 5.0})
        pool = _pool_with({"g": 2.0, "x": 1.0}, ["a"])
        assert coverage_stratum(pool, ppmi) == "distractor"

    def test_none_when_the_graph_is_silent_or_there_are_no_anchors(self):
        from locus.eval.rank import coverage_stratum
        pool = _pool_with({"g": 2.0, "x": 1.0}, ["a"])
        assert coverage_stratum(pool, _TablePPMI({})) == "none"
        empty = _pool_with({"g": 2.0, "x": 1.0}, [])
        assert coverage_stratum(empty, _TablePPMI({("g", "a"): 5.0})) == "none"

    def test_stratum_uses_the_requested_aggregator(self):
        from locus.eval.rank import coverage_stratum
        ppmi = _RecordingPPMI(value=0.0)
        coverage_stratum(_pool_with({"g": 1.0, "x": 2.0}, ["a"]), ppmi,
                         agg="masked_mean")
        assert set(ppmi.seen) == {"masked_mean"}


def test_coverage_strata_partition_the_pools_and_sign_their_deltas():
    # One pool per stratum, each from its own citing paper. The graph can only
    # move the gold where it saw it (up), only move a distractor where it saw
    # one (gold down), and cannot move anything where it saw nothing.
    from locus.eval.rank import COVERAGE_STRATA, coverage_strata
    ppmi = _TablePPMI({("g", "a"): 5.0, ("x", "b"): 5.0})
    pools = [
        _pool_with({"g": 1.0, "x": 2.0}, ["a"], citing="p1", ctx="c1"),   # target
        _pool_with({"g": 2.0, "x": 1.0}, ["b"], citing="p2", ctx="c2"),   # distractor
        _pool_with({"g": 2.0, "x": 1.0}, ["z"], citing="p3", ctx="c3"),   # none
    ]
    got = coverage_strata(pools, ppmi, lam=1.0, agg="mean", seed=0)
    assert COVERAGE_STRATA == ("target", "distractor", "none")
    assert list(got)[:3] == list(COVERAGE_STRATA)   # then the combined row
    assert [got[s]["n"] for s in COVERAGE_STRATA] == [1, 1, 1]
    assert sum(got[s]["share"] for s in COVERAGE_STRATA) == pytest.approx(1.0)
    assert got["target"]["delta"] == pytest.approx(0.5)
    assert got["distractor"]["delta"] == pytest.approx(-0.5)
    assert got["none"]["delta"] == 0.0
    for s in COVERAGE_STRATA:
        assert {"n", "share", "a", "b", "delta", "ci"} <= set(got[s])


def test_coverage_strata_also_reports_the_combined_unseen_row():
    # The paper's coverage table has a PPMI(c, A) = 0 row that is the union of
    # `distractor` and `none`, with its own paired interval. It is reported
    # alongside the partition, not instead of it, and its count is the sum.
    from locus.eval.rank import COVERAGE_STRATA, coverage_strata
    ppmi = _TablePPMI({("g", "a"): 5.0, ("x", "b"): 5.0})
    pools = [
        _pool_with({"g": 1.0, "x": 2.0}, ["a"], citing="p1", ctx="c1"),
        _pool_with({"g": 2.0, "x": 1.0}, ["b"], citing="p2", ctx="c2"),
        _pool_with({"g": 2.0, "x": 1.0}, ["z"], citing="p3", ctx="c3"),
        _pool_with({"g": 2.0, "x": 1.0}, ["z"], citing="p4", ctx="c4"),
    ]
    got = coverage_strata(pools, ppmi, lam=1.0, agg="mean", seed=0)
    assert list(got)[:3] == list(COVERAGE_STRATA)
    assert got["unseen"]["n"] == got["distractor"]["n"] + got["none"]["n"] == 3
    assert got["unseen"]["share"] == pytest.approx(0.75)
    assert got["unseen"]["delta"] == pytest.approx(-0.5 / 3)


def test_main_reports_coverage_and_the_other_location_arm_against_base(
        monkeypatch, tmp_path):
    # Table 7's "other location" row is stated against the BASE (0.478 ->
    # 0.388), while gate 2 pairs it against the same-location arm. Both
    # readings come from one run, so the report has to carry both intervals.
    import json

    ppmi = _TablePPMI({("g", "a"): 5.0, ("x", "b"): 5.0})
    pools = [
        _pool_with({"g": 1.0, "x": 2.0}, ["a"], citing="p1", ctx="c1"),
        _pool_with({"g": 2.0, "x": 1.0}, ["b"], citing="p2", ctx="c2"),
        _pool_with({"g": 2.0, "x": 1.0}, ["z"], citing="p3", ctx="c3"),
    ]
    control = {"c1": frozenset(["b"]), "c2": frozenset(["a"]), "c3": frozenset(["a"])}
    monkeypatch.setattr(rank_mod, "build_pools", lambda *a, **k: pools)
    monkeypatch.setattr(rank_mod, "other_location_anchors", lambda split, seed: control)
    monkeypatch.setattr(rank_mod, "base_scorer", lambda *a, **k: None)
    monkeypatch.setattr(rank_mod.CoGraph, "load", classmethod(lambda cls, p: None))
    monkeypatch.setattr(rank_mod, "association", lambda *a, **k: ppmi)
    sweep = tmp_path / "sweep.json"
    sweep.write_text(json.dumps({"best": {"alpha": 0.5, "min_count": 1,
                                          "lambda": 1.0}}))
    out = tmp_path / "report.json"
    rank_mod.main(["--embeddings", "x.parquet", "--sweep", str(sweep),
               "--graph", "g.npz", "--out", str(out)])
    report = json.loads(out.read_text())

    assert set(report["coverage_strata"]) == {"target", "distractor", "none", "unseen"}
    row = report["other_location_vs_base"]
    assert {"a", "b", "delta", "ci", "n"} <= set(row)
    assert row["b"] == pytest.approx(report["gate1_primary"]["b"])
    assert row["a"] == pytest.approx(report["gate2_other_location_control"]["b"])
