"""Sensitivity of the headline to the location-clustering threshold.

Changing theta changes three things at once -- which contexts are eligible,
which nine distractors each pool holds, and which anchors each context has --
so a delta read across thresholds is uninterpretable unless the first two are
held fixed. These tests pin the cohort logic that holds them fixed.
"""
import pytest

from locus.experiments.theta import cohorts, summarise


def _row(cands, rr_base, rr_resc, citing="p", n_anchors=1):
    return {"citing_id": citing, "candidates": tuple(cands), "base": rr_base,
            "rescored": rr_resc, "n_anchors": n_anchors}


PER_THETA = {
    0.4: {"c1": _row("abc", 0.5, 1.0), "c2": _row("abd", 0.5, 0.5), "c3": _row("xyz", 1.0, 1.0)},
    0.5: {"c1": _row("abc", 0.5, 1.0), "c2": _row("abe", 0.5, 1.0), "c4": _row("xyz", 1.0, 1.0)},
    0.6: {"c1": _row("abc", 0.5, 0.5), "c2": _row("abe", 0.5, 1.0), "c4": _row("xyz", 1.0, 1.0)},
}


def test_the_fixed_cohort_is_the_contexts_eligible_at_every_threshold():
    common, _same_pool = cohorts(PER_THETA)
    assert common == {"c1", "c2"}          # c3 and c4 drop in and out


def test_the_fixed_pool_cohort_additionally_requires_identical_candidates():
    # c2's distractors change between 0.4 and 0.5, so a delta difference on it
    # could be the pool rather than the anchors. Only c1 isolates the anchors.
    _common, same_pool = cohorts(PER_THETA)
    assert same_pool == {"c1"}


def test_summarise_restricts_to_the_cohort_and_reports_both_averaging_units():
    got = summarise(PER_THETA[0.5], keep={"c1", "c2"}, seed=0)
    assert got["n"] == 2 and got["papers"] == 1
    assert got["b"] == pytest.approx(0.5) and got["a"] == pytest.approx(1.0)
    assert got["pool_delta"] == pytest.approx(0.5)
    assert got["mean_anchors"] == 1.0


def test_summarise_without_a_cohort_uses_every_pool():
    got = summarise(PER_THETA[0.5], keep=None, seed=0)
    assert got["n"] == 3


# ---- holding the pools fixed while the anchors move --------------------------
#
# A pool's distractors are sampled from the references OUTSIDE the location's
# cited set, so a pool only survives a change of threshold when that set does
# -- and then the anchors have not moved either. "Same pool at every theta"
# therefore cannot isolate the anchors. The reading that can keeps the
# reference pools and swaps in the anchors each threshold infers.

class _Assoc:
    """PPMI stand-in: one unit of association between named pairs."""
    def __init__(self, pairs):
        self._pairs = {frozenset(p) for p in pairs}

    def mean_over(self, c, anchors):
        if not anchors:
            return 0.0
        return sum(frozenset((c, a)) in self._pairs for a in anchors) / len(anchors)


REF = {
    "c1": {"citing_id": "p", "gold": "g", "candidates": ("g", "d"),
           "base_scores": {"g": 0.1, "d": 0.2}},
    "c2": {"citing_id": "q", "gold": "g", "candidates": ("g", "d"),
           "base_scores": {"g": 0.1, "d": 0.2}},
    "c3": {"citing_id": "q", "gold": "g", "candidates": ("g", "d"),
           "base_scores": {"g": 0.1, "d": 0.2}},
}


def test_the_swap_cohort_needs_usable_anchors_at_every_threshold():
    from locus.experiments.theta import swap_cohort

    anchors = {
        0.3: {"c1": frozenset({"a"}), "c2": frozenset({"a", "d"}), "c3": frozenset({"a"})},
        0.5: {"c1": frozenset({"a"}), "c2": frozenset({"a"}), "c3": frozenset()},
    }
    # c2: at 0.3 a candidate became an anchor, so the pool is no longer a
    # clean test of that anchor set. c3: no anchor at 0.5.
    assert swap_cohort(REF, anchors) == {"c1"}


def test_swapped_anchors_change_the_score_and_nothing_else():
    from locus.experiments.theta import summarise_swapped

    assoc = _Assoc([("g", "a")])
    helpful = summarise_swapped(REF, {"c1": frozenset({"a"})}, assoc, lam=5.0,
                                keep={"c1"}, seed=0)
    useless = summarise_swapped(REF, {"c1": frozenset({"z"})}, assoc, lam=5.0,
                                keep={"c1"}, seed=0)
    assert helpful["n"] == useless["n"] == 1
    assert helpful["b"] == useless["b"] == pytest.approx(0.5)   # same pool, same base
    assert helpful["a"] == pytest.approx(1.0)                   # the anchor lifts the target
    assert useless["a"] == pytest.approx(0.5)
    assert helpful["mean_anchors"] == 1.0


def test_the_fixed_pool_cohort_ignores_candidate_order():
    # The same ten papers in another order are the same pool.
    per_theta = {0.4: {"c1": _row("abc", 0.5, 1.0)}, 0.5: {"c1": _row("cab", 0.5, 1.0)}}
    assert cohorts(per_theta)[1] == {"c1"}
