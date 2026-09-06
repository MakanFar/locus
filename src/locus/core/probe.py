"""LOCUS-Rank and LOCUS-Swap scoring rules.

Ties are scored by their exact expectation under random tie-breaking, never
broken randomly and never by fractional rank. Both conventions are in
circulation and they disagree: for a k-way tie, E[1/rank] = H_k/k while
1/E[rank] = 2/(k+1) -- 0.29290 against 0.18182 at k=10. Reporting R@1 under
one and MRR under the other would put two different conventions in the same
table, so both metrics here are the expectation of the randomised procedure
they stand in for. That keeps them exact and zero-variance: a scorer that
ignores the context produces a bit-exact zero Swap delta (the same two floats
added in a different order), pinning it to exactly 0.5, and ties every Rank
pool, pinning MRR to exactly H_10/10.
"""


def _band(scores: dict[str, float], gold: str) -> tuple[int, int]:
    """Return (candidates strictly above the gold, candidates tied with it).

    The gold counts itself in `tied`, so `tied >= 1` always.
    """
    g = scores[gold]
    if g != g or any(v != v for v in scores.values()):
        # NaN compares false against everything, so a NaN gold would be
        # counted neither above nor tied and score 1/0.5 = 2.0, above the
        # ceiling; NaN distractors would vanish from the pool and make a
        # scorer that fails on hard candidates look perfect. Swap already
        # refuses NaN; Rank is the pre-registered primary metric and must
        # refuse it at least as loudly.
        raise ValueError(
            f"NaN score in a rank pool (gold={gold!r}): {scores!r}; a scorer "
            "emitting NaN would otherwise report above the metric ceiling"
        )
    above = sum(1 for v in scores.values() if v > g)
    tied = sum(1 for v in scores.values() if v == g)
    return above, tied


def reciprocal_rank(scores: dict[str, float], gold: str) -> float:
    """E[1/rank] with the gold's rank uniform over its tied band."""
    above, tied = _band(scores, gold)
    return sum(1.0 / i for i in range(above + 1, above + tied + 1)) / tied


def hit_at_1(scores: dict[str, float], gold: str) -> float:
    """P(rank == 1), i.e. E[hit@1] over random tie-breaking."""
    above, tied = _band(scores, gold)
    if above:
        return 0.0
    return 1.0 / tied


def swap_outcome(
    s_a_ga: float,
    s_b_gb: float,
    s_a_gb: float,
    s_b_ga: float,
) -> float:
    delta = (s_a_ga + s_b_gb) - (s_a_gb + s_b_ga)
    if delta != delta:  # NaN
        raise ValueError(
            "swap_outcome received a NaN score: "
            f"s_a_ga={s_a_ga!r} s_b_gb={s_b_gb!r} s_a_gb={s_a_gb!r} s_b_ga={s_b_ga!r}; "
            "a scorer emitting NaN would otherwise report as a total loss on "
            "every pair"
        )
    # The zero comparison is exact and deliberate, not a numerical-tolerance
    # shortcut: for a context-independent scorer, s_a_ga+s_b_gb and
    # s_a_gb+s_b_ga are the same two floats added in the opposite order, and
    # IEEE-754 addition is commutative, so delta is bit-exactly 0.0. Any
    # epsilon here would mask real harness bugs inside the tolerance band.
    if delta == 0.0:
        return 0.5
    return 1.0 if delta > 0 else 0.0
