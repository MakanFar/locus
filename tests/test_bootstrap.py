import pytest

from locus.core.bootstrap import cluster_ci, paper_mean


class TestClusterCI:
    def test_zero_variance_gives_degenerate_interval(self):
        lo, hi = cluster_ci([0.5] * 200, [f"p{i // 4}" for i in range(200)], n_boot=200, seed=0)
        assert lo == 0.5 and hi == 0.5

    def test_interval_brackets_the_mean(self):
        vals = [0.0, 1.0] * 100
        groups = [f"p{i // 4}" for i in range(200)]
        lo, hi = cluster_ci(vals, groups, n_boot=500, seed=0)
        assert lo <= 0.5 <= hi

    def test_deterministic_under_seed(self):
        vals = [i % 3 / 2 for i in range(120)]
        groups = [f"p{i // 3}" for i in range(120)]
        assert cluster_ci(vals, groups, n_boot=200, seed=4) == cluster_ci(
            vals, groups, n_boot=200, seed=4
        )

    def test_resamples_papers_not_rows(self):
        # One paper carries every 1.0. Resampling papers must sometimes drop it
        # entirely, so the lower bound reaches 0.0; row-resampling never would.
        vals = [1.0] * 50 + [0.0] * 50
        groups = ["hot"] * 50 + [f"p{i}" for i in range(50)]
        lo, _ = cluster_ci(vals, groups, n_boot=1000, seed=0)
        assert lo == 0.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            cluster_ci([1.0, 2.0, 3.0], ["p0", "p1"])


class TestPaperMean:
    def test_each_paper_counts_once(self):
        # Mean-over-pairs and mean-over-papers are different metrics, and
        # only the latter has correct CIs. One reference-dense
        # paper contributing 99 pairs must not outweigh 99 papers with one.
        vals = [1.0] + [0.0] * 99
        groups = ["solo"] + ["dense"] * 99
        assert sum(vals) / len(vals) == pytest.approx(0.01)  # pair-weighted
        assert paper_mean(vals, groups) == pytest.approx(0.5)  # paper-weighted

    def test_balanced_input_agrees_with_the_pair_mean(self):
        vals = [1.0, 0.0, 1.0, 0.0]
        groups = ["a", "a", "b", "b"]
        assert paper_mean(vals, groups) == pytest.approx(0.5)

    def test_empty_is_nan(self):
        import math

        assert math.isnan(paper_mean([], []))

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            paper_mean([1.0, 2.0, 3.0], ["p0", "p1"])


class TestClusterCIIsPaperWeighted:
    def test_interval_centres_on_the_paper_mean_not_the_pair_mean(self):
        # 1 paper of 1 pair scoring 1.0, 1 paper of 99 pairs scoring 0.0.
        # Pair-weighted mean is 0.01; paper-weighted is 0.5. Resampling two
        # papers with replacement gives means {1.0, 0.5, 0.5, 0.0}, so the
        # median of the bootstrap distribution is 0.5, not 0.01.
        import numpy as np

        vals = [1.0] + [0.0] * 99
        groups = ["solo"] + ["dense"] * 99
        lo, hi = cluster_ci(vals, groups, n_boot=2000, seed=0)
        assert lo == 0.0 and hi == 1.0
        assert np.mean([lo, hi]) == pytest.approx(0.5)

    def test_null_control_still_collapses_to_a_point(self):
        # The exact-0.5 guarantee must survive the reweighting.
        lo, hi = cluster_ci([0.5] * 200, [f"p{i // 7}" for i in range(200)], n_boot=200, seed=0)
        assert lo == 0.5 and hi == 0.5


class TestExactnessOfTheNullFloors:
    """The harness's central bet is that a null control lands on its floor
    *exactly*, with a zero-width interval, so "50% +/- noise" becomes an
    assertion. That survives only if aggregation is exactly rounded --
    numpy's pairwise summation is not, and drifts by an ULP at these sizes."""

    FLOOR = sum(1 / i for i in range(1, 11)) / 10  # H_10/10, the Rank tie floor

    def test_exact_when_averaging_across_many_papers(self):
        n = 1000  # one item per paper; np.mean over 1000 copies is off by 1 ULP
        assert paper_mean([self.FLOOR] * n, [f"p{i}" for i in range(n)]) == self.FLOOR

    def test_exact_when_averaging_within_one_dense_paper(self):
        n = 99  # np.mean over 99 copies is off by 1 ULP
        assert paper_mean([self.FLOOR] * n, ["dense"] * n) == self.FLOOR

    def test_exact_at_the_frozen_headline_size(self):
        n = 38_826
        assert paper_mean([self.FLOOR] * n, [f"p{i}" for i in range(n)]) == self.FLOOR

    def test_cluster_ci_is_zero_width_at_the_rank_floor(self):
        n = 1000
        lo, hi = cluster_ci([self.FLOOR] * n, [f"p{i}" for i in range(n)], n_boot=200)
        assert lo == self.FLOOR and hi == self.FLOOR

    def test_the_swap_floor_too(self):
        n = 27_905  # the size of the frozen swap set
        groups = [f"p{i // 3}" for i in range(n)]
        assert paper_mean([0.5] * n, groups) == 0.5
        assert cluster_ci([0.5] * n, groups, n_boot=200) == (0.5, 0.5)
