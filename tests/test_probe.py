import pytest

from locus.core.probe import hit_at_1, reciprocal_rank, swap_outcome


class TestReciprocalRank:
    def test_top_ranked_gold(self):
        assert reciprocal_rank({"g": 9.0, "a": 1.0, "b": 0.5}, "g") == 1.0

    def test_third_ranked_gold(self):
        assert reciprocal_rank({"g": 0.1, "a": 1.0, "b": 0.5}, "g") == pytest.approx(1 / 3)

    def test_all_tied_gives_expected_reciprocal_rank(self):
        # A k-way tie means the gold's rank is uniform on 1..k, so the
        # unbiased reciprocal rank is E[1/rank] = H_k / k, NOT 1/E[rank] =
        # 2/(k+1). For k=10 that is 0.29290, not 0.18182. Using 1/E[rank]
        # here while hit_at_1 uses E[hit] would make the two metrics report
        # different conventions for the same tie.
        scores = {k: 0.0 for k in "abcdefghij"}
        expected = sum(1 / i for i in range(1, 11)) / 10
        assert reciprocal_rank(scores, "a") == pytest.approx(expected)

    def test_partial_tie_at_the_top(self):
        # g tied with one other at the top -> rank is 1 or 2 with equal
        # probability -> E[1/rank] = (1/1 + 1/2)/2 = 0.75.
        assert reciprocal_rank({"g": 1.0, "a": 1.0, "b": 0.0}, "g") == pytest.approx(0.75)

    def test_tie_below_the_top_uses_the_tied_band(self):
        # one strictly above, then a 3-way tie -> rank uniform on 2..4.
        scores = {"top": 9.0, "g": 1.0, "a": 1.0, "b": 1.0}
        expected = (1 / 2 + 1 / 3 + 1 / 4) / 3
        assert reciprocal_rank(scores, "g") == pytest.approx(expected)

    def test_matches_monte_carlo_random_tie_breaking(self):
        # The convention is not a matter of taste: it must equal the mean of
        # the randomised procedure it stands in for.
        import random
        import statistics

        scores = {"top": 9.0, "g": 1.0, "a": 1.0, "b": 1.0, "c": 1.0, "low": 0.0}
        rng = random.Random(0)
        trials = []
        for _ in range(200_000):
            order = sorted(scores, key=lambda k: (-scores[k], rng.random()))
            trials.append(1 / (order.index("g") + 1))
        assert reciprocal_rank(scores, "g") == pytest.approx(
            statistics.fmean(trials), abs=2e-3
        )

    def test_nan_gold_raises(self):
        # NaN compares false against everything, so the gold would be counted
        # neither above nor tied and score 1/0.5 = 2.0 -- above the ceiling.
        with pytest.raises(ValueError):
            reciprocal_rank({"g": float("nan"), "a": 1.0, "b": 2.0}, "g")

    def test_nan_distractor_raises(self):
        # A scorer that NaNs on hard candidates would otherwise look perfect.
        with pytest.raises(ValueError):
            reciprocal_rank({"g": 0.0, "a": float("nan"), "b": float("nan")}, "g")


class TestHitAt1:
    def test_clear_win(self):
        assert hit_at_1({"g": 9.0, "a": 1.0}, "g") == 1.0

    def test_clear_loss(self):
        assert hit_at_1({"g": 0.0, "a": 1.0}, "g") == 0.0

    def test_two_way_tie_gives_half(self):
        assert hit_at_1({"g": 1.0, "a": 1.0, "b": 0.0}, "g") == pytest.approx(0.5)

    def test_nan_gold_raises(self):
        with pytest.raises(ValueError):
            hit_at_1({"g": float("nan"), "a": 1.0}, "g")

    def test_nan_distractor_raises(self):
        with pytest.raises(ValueError):
            hit_at_1({"g": 1.0, "a": float("nan")}, "g")


class TestSwapOutcome:
    def test_true_assignment_wins(self):
        assert swap_outcome(5.0, 5.0, 1.0, 1.0) == 1.0

    def test_swapped_assignment_wins(self):
        assert swap_outcome(1.0, 1.0, 5.0, 5.0) == 0.0

    def test_exact_tie_scores_half(self):
        assert swap_outcome(2.0, 3.0, 3.0, 2.0) == 0.5

    def test_context_independent_scorer_is_exactly_half(self):
        # s depends only on the document: s(l1,G1)=f(G1), s(l2,G2)=f(G2),
        # s(l1,G2)=f(G2), s(l2,G1)=f(G1). Commutativity of float addition
        # makes delta bit-exactly zero, so this is an exact assertion.
        import random

        rng = random.Random(0)
        for _ in range(10_000):
            f_g1, f_g2 = rng.uniform(-1e6, 1e6), rng.uniform(-1e6, 1e6)
            assert swap_outcome(f_g1, f_g2, f_g2, f_g1) == 0.5

    def test_mean_over_context_independent_pairs_is_exactly_half(self):
        import random

        rng = random.Random(1)
        outs = []
        for _ in range(5_000):
            a, b = rng.uniform(0, 1), rng.uniform(0, 1)
            outs.append(swap_outcome(a, b, b, a))
        assert sum(outs) / len(outs) == 0.5  # exact, not approx

    def test_nan_score_raises(self):
        with pytest.raises(ValueError):
            swap_outcome(float("nan"), 1.0, 1.0, 1.0)

    def test_nan_error_names_all_four_operands(self):
        with pytest.raises(ValueError, match=r"2\.0.*3\.0.*4\.0.*nan"):
            swap_outcome(2.0, 3.0, 4.0, float("nan"))
