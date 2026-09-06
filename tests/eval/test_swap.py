import math

import pytest

from locus.core.rescore import zscore_pool
from locus.eval.swap import accuracy, global_stats


class _PPMI:
    """Stub with the one method swap.py calls."""

    def __init__(self, table=None):
        self.table = table or {}

    def mean_over(self, c, anchors):
        return self.table.get((c, frozenset(anchors)), 0.0)


def _pair(citing, base, ga="ga", gb="gb"):
    return {"citing_id": citing, "gold_a": ga, "gold_b": gb,
            "anch_a": frozenset({"x"}), "anch_b": frozenset({"y"}),
            "base": base}


# p1's two pairs both win and p2's single pair loses, so the paper mean (0.5)
# and the pair mean (2/3) differ -- with a symmetric fixture the two
# weightings coincide and the test cannot see the difference.
PAIRS = [
    _pair("p1", (0.90, 0.80, 0.10, 0.20)),   # true assignment wins
    _pair("p1", (0.70, 0.60, 0.20, 0.30)),   # true assignment wins
    _pair("p2", (0.10, 0.20, 0.90, 0.80)),   # swapped wins
]


def test_global_mean_cancels_exactly():
    # delta has two positive and two negative base terms, so any shift applied
    # uniformly leaves the outcome untouched. This is what lets sigma be the
    # only meaningful constant -- and it must hold exactly, not approximately.
    ppmi = _PPMI()
    a, _, _ = accuracy(PAIRS, ppmi, 0.0, mu=0.0, sigma=1.0, arm="base")
    for mu in (-1e6, -3.7, 0.25, 1e6):
        b, _, _ = accuracy(PAIRS, ppmi, 0.0, mu=mu, sigma=1.0, arm="base")
        assert b == a


def test_sigma_does_not_change_the_base_only_arm():
    # With no PPMI term, sigma is a positive rescale of delta and cannot flip
    # its sign -- so it only bites once the two signals are being traded off.
    ppmi = _PPMI()
    a, _, _ = accuracy(PAIRS, ppmi, 0.0, mu=0.0, sigma=1.0, arm="base")
    for sigma in (0.01, 0.5, 17.0):
        b, _, _ = accuracy(PAIRS, ppmi, 0.0, mu=0.0, sigma=sigma, arm="base")
        assert b == a


def test_sigma_sets_the_exchange_rate_once_ppmi_is_added():
    # The whole reason the spec forbids pool-z here: with a 2-candidate pool,
    # z-scoring pins the base contrast to a constant and sigma is what
    # actually decides how much the co-citation term can override it.
    ppmi = _PPMI({("gb", frozenset({"x"})): 1.0})   # evidence for the WRONG answer
    pairs = [_pair("p1", (0.90, 0.80, 0.10, 0.20))]
    strong_base, _, _ = accuracy(pairs, ppmi, 1.0, mu=0.0, sigma=0.01, arm="both")
    weak_base, _, _ = accuracy(pairs, ppmi, 1.0, mu=0.0, sigma=100.0, arm="both")
    assert strong_base == 1.0   # base dominates, true assignment survives
    assert weak_base == 0.0     # base flattened, the misleading PPMI wins


def test_global_scaling_beats_pool_z_where_confidence_is_uneven():
    # The case that separates them. Context a prefers its own gold strongly
    # (0.90 vs 0.10); context b prefers the WRONG gold, but barely (0.55 vs
    # 0.50). Summed globally the true assignment wins by +0.75. Under pool-z
    # each context contributes a bare +1/-1 vote, the margins vanish, and the
    # pair lands on an artificial tie.
    ppmi = _PPMI()
    uneven = [_pair("p1", (0.90, 0.50, 0.10, 0.55))]
    _acc, outcomes, _ = accuracy(uneven, ppmi, 0.0, mu=0.0, sigma=1.0, arm="base")
    assert outcomes == [1.0]
    per_pool = [1.0, -1.0, -1.0, 1.0]   # what zscore_pool gives each context
    delta = (per_pool[0] + per_pool[1]) - (per_pool[2] + per_pool[3])
    assert delta == 0.0


def test_pool_zscoring_would_destroy_the_two_candidate_contrast():
    # Why this module does not reuse rescore.zscore_pool: on two candidates it
    # returns the same two numbers regardless of input, so every pair would
    # carry identical base evidence.
    assert sorted(zscore_pool({"a": 0.9, "b": 0.1}).values()) == [-1.0, 1.0]
    assert sorted(zscore_pool({"a": 0.51, "b": 0.50}).values()) == [-1.0, 1.0]


def test_exact_ties_score_half():
    ppmi = _PPMI()
    tied = [_pair("p1", (0.5, 0.5, 0.5, 0.5))]
    _, outcomes, _ = accuracy(tied, ppmi, 0.0, mu=0.0, sigma=1.0, arm="base")
    assert outcomes == [0.5]


def test_paper_weighted_not_pair_weighted():
    # p1 contributes two wins, p2 one loss: paper mean (1.0 + 0.0)/2 = 0.5,
    # against a pair mean of 2/3.
    ppmi = _PPMI()
    acc, outcomes, _ = accuracy(PAIRS, ppmi, 0.0, mu=0.0, sigma=1.0, arm="base")
    assert outcomes == [1.0, 1.0, 0.0]
    assert acc == pytest.approx(0.5)
    assert acc != pytest.approx(2 / 3)


def test_global_stats_uses_every_score_and_is_exactly_summed():
    mu, sigma = global_stats(PAIRS)
    vals = [v for p in PAIRS for v in p["base"]]
    assert mu == math.fsum(vals) / len(vals)
    assert sigma == pytest.approx(
        math.sqrt(math.fsum((v - mu) ** 2 for v in vals) / len(vals))
    )
