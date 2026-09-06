import random

from locus.core.pools import RankItem, SwapPair
from locus.scoring.baselines import (
    constant_scorer,
    popularity_scorer,
    rank_mrr,
    swap_accuracy,
)


def _pairs(n):
    return [
        SwapPair(
            citing_id=f"P{i // 3}",
            ctx_a=f"P{i // 3}_a{i}",
            ctx_b=f"P{i // 3}_b{i}",
            gold_a=f"g{2 * i}",
            gold_b=f"g{2 * i + 1}",
        )
        for i in range(n)
    ]


class TestNullControls:
    def test_popularity_is_exactly_half_on_swap(self):
        pairs = _pairs(300)
        rng = random.Random(0)
        counts = {}
        for p in pairs:
            counts[p.gold_a] = rng.randint(1, 10_000)
            counts[p.gold_b] = rng.randint(1, 10_000)
        acc, outcomes, groups = swap_accuracy(pairs, popularity_scorer(counts))
        assert acc == 0.5                      # exact, by the lemma
        assert set(outcomes) == {0.5}
        assert len(groups) == len(pairs)

    def test_constant_scorer_is_exactly_half_on_swap(self):
        acc, outcomes, _ = swap_accuracy(_pairs(100), constant_scorer(3.7))
        assert acc == 0.5
        assert set(outcomes) == {0.5}

    def test_a_context_dependent_scorer_can_leave_half(self):
        # sanity: the harness is capable of producing something other than 0.5,
        # so the two assertions above are testing the lemma and not a stuck value
        pairs = _pairs(50)

        def cheating(context_id: str, candidate_id: str) -> float:
            # rewards the true assignment: gold_a index is even, ctx_a marked 'a'
            return 1.0 if ("_a" in context_id) == (int(candidate_id[1:]) % 2 == 0) else 0.0

        acc, _, _ = swap_accuracy(pairs, cheating)
        assert acc == 1.0


class TestSwapAccuracyIsPaperWeighted:
    def test_a_dense_paper_does_not_dominate(self):
        # One paper with 99 pairs the scorer always loses, one paper with a
        # single pair it always wins. Pair-weighted -> 0.01. Paper-weighted
        # -> 0.5.
        dense = [
            SwapPair(citing_id="dense", ctx_a=f"d_a{i}", ctx_b=f"d_b{i}",
                     gold_a="lose", gold_b="win")
            for i in range(99)
        ]
        solo = [SwapPair(citing_id="solo", ctx_a="s_a", ctx_b="s_b",
                         gold_a="win", gold_b="lose")]

        def scorer(context_id: str, candidate_id: str) -> float:
            # true assignment beats swapped for solo, loses for dense
            return 1.0 if (candidate_id == "win") == ("_a" in context_id) else 0.0

        acc, outcomes, _ = swap_accuracy(dense + solo, scorer)
        assert sum(outcomes) / len(outcomes) < 0.02   # the pair-weighted reading
        assert acc == 0.5                              # the reported number

    def test_null_control_survives_the_reweighting(self):
        acc, outcomes, _ = swap_accuracy(_pairs(300), constant_scorer(3.7))
        assert acc == 0.5
        assert set(outcomes) == {0.5}


def _items(n, pool=10):
    return [
        RankItem(
            context_id=f"P{i // 3}_c{i}",
            citing_id=f"P{i // 3}",
            gold=f"g{i}_0",
            candidates=tuple(f"g{i}_{j}" for j in range(pool)),
        )
        for i in range(n)
    ]


class TestRankNullControl:
    def test_constant_scorer_gives_the_exact_tie_floor(self):
        # Every candidate ties, so MRR is the closed-form H_10/10 exactly --
        # the same kind of assertion the Swap lemma gives, on the metric the
        # spec pre-registers as primary.
        mrr, values, groups = rank_mrr(_items(300), constant_scorer(1.0))
        floor = sum(1 / i for i in range(1, 11)) / 10
        assert set(values) == {floor}
        assert mrr == floor
        assert len(groups) == 300

    def test_popularity_is_not_pinned_to_the_floor(self):
        # popularity varies across candidates, so unlike Swap it is a real
        # (weak) baseline on Rank, not an identity. This is the guard that the
        # floor above is the tie rule and not a stuck value.
        items = _items(50)
        counts = {c: int(c.split("_")[1]) for it in items for c in it.candidates}
        mrr, _, _ = rank_mrr(items, popularity_scorer(counts))
        assert mrr != sum(1 / i for i in range(1, 11)) / 10

    def test_is_paper_weighted(self):
        dense = [
            RankItem(context_id=f"d{i}", citing_id="dense", gold="a",
                     candidates=("a", "b"))
            for i in range(99)
        ]
        solo = [RankItem(context_id="s", citing_id="solo", gold="a",
                         candidates=("a", "b"))]

        def scorer(context_id: str, candidate_id: str) -> float:
            # gold wins only for the solo paper
            return 1.0 if (candidate_id == "a") == context_id.startswith("s") else 0.0

        mrr, values, _ = rank_mrr(dense + solo, scorer)
        assert sum(values) / len(values) < 0.51   # pair-weighted reading
        assert mrr == 0.75                        # (0.5 + 1.0) / 2
