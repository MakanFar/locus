import pytest

from locus.core.rescore import zscore_pool
from locus.eval.sweep import mrr_over, mrr_rescored


def _pool(cid, paper, gold, base):
    return {
        "context_id": cid, "citing_id": paper, "gold": gold,
        "base": base, "z": zscore_pool(base), "anchors": frozenset(),
    }


POOLS = [
    _pool("c1", "p1", "g", {"g": 0.30, "x": 0.90, "y": 0.10}),
    _pool("c2", "p1", "g", {"g": 0.80, "x": 0.20, "y": 0.55}),
    _pool("c3", "p2", "g", {"g": 0.05, "x": 0.06, "y": 0.99}),
]


def test_zscoring_leaves_mrr_exactly_unchanged():
    # Strictly monotone within a pool, so not merely close: equal. The sweep
    # refuses to select a lambda if this ever fails, because it would mean the
    # rescoring path perturbs the base ranking it is supposed to add to.
    assert mrr_over(POOLS, lambda p: p["base"]) == mrr_over(POOLS, lambda p: p["z"])


def test_lambda_zero_reproduces_the_base_exactly():
    cached = [dict.fromkeys(p["base"], 0.7) for p in POOLS]  # nonzero on purpose
    assert mrr_rescored(POOLS, cached, 0.0) == mrr_over(POOLS, lambda p: p["base"])


def test_a_uniform_ppmi_shift_cannot_change_the_ranking():
    # Adding the same constant to every candidate in a pool is rank-preserving
    # at any lambda; only *differences* between candidates can move MRR.
    cached = [dict.fromkeys(p["base"], 0.7) for p in POOLS]
    for lam in (0.1, 1.0, 5.0):
        assert mrr_rescored(POOLS, cached, lam) == mrr_over(POOLS, lambda p: p["base"])


def test_ppmi_evidence_on_the_gold_lifts_mrr():
    # The gold is bottom of pool c3 on cosine alone; enough co-citation
    # evidence must be able to rescue it, or the rescorer cannot help at all.
    cached = [{c: (1.0 if c == p["gold"] else 0.0) for c in p["base"]} for p in POOLS]
    assert mrr_rescored(POOLS, cached, 5.0) > mrr_over(POOLS, lambda p: p["base"])


def test_paper_weighting_not_pool_weighting():
    # p1 contributes two pools and p2 one. A pool-weighted mean would be the
    # mean of three reciprocal ranks; the reported convention is the mean of the
    # two per-paper means.
    got = mrr_over(POOLS, lambda p: p["base"])
    p1 = (1 / 2 + 1 / 1) / 2   # c1 gold ranks 2nd, c2 gold ranks 1st
    p2 = 1 / 3                 # c3 gold ranks 3rd
    assert got == pytest.approx((p1 + p2) / 2)


def test_build_pools_rejects_an_unknown_slice():
    # The check has to fire BEFORE any artefact is opened, or a typo'd slice
    # reads 300 MB of pickles and then fails.
    from pathlib import Path

    from locus.eval.sweep import build_pools
    with pytest.raises(ValueError, match="unknown slice"):
        build_pools("test", Path("/nonexistent"), "hard")
