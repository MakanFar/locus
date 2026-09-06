"""Paper-level cluster bootstrap.

Contexts from one citing paper are not independent, so resampling rows would
understate the interval. Resample citing papers with replacement and take all
their rows.

Weighting is per paper, not per row: mean-over-pairs and mean-over-papers
are different metrics, and only the latter has correct CIs under a paper-level
resample. Resampling the paper while still averaging over rows
would leave a reference-dense paper contributing 5 swap pairs outweighing five
papers contributing one each -- exactly the imbalance the k=5 cap bounds but
does not remove.
"""
import collections
import math

import numpy as np


def _mean(xs: list[float]) -> float:
    # math.fsum, not sum() or np.mean: the null controls must land on their
    # floor *exactly* (0.5 on Swap, H_10/10 on Rank), and numpy's pairwise
    # summation drifts by an ULP once there are ~100 terms, which would turn
    # an exact assertion back into an approximate one.
    return math.fsum(xs) / len(xs)


def _paper_means(values: list[float], groups: list[str]) -> tuple[list[str], np.ndarray]:
    buckets: dict[str, list[float]] = collections.defaultdict(list)
    for v, g in zip(values, groups, strict=True):
        buckets[g].append(v)
    keys = sorted(buckets)
    return keys, np.array([_mean(buckets[k]) for k in keys], dtype=float)


def paper_mean(values: list[float], groups: list[str]) -> float:
    """The point estimate that goes with `cluster_ci`: mean over papers."""
    if not values:
        return float("nan")
    _, means = _paper_means(values, groups)
    return _mean(means.tolist())


def cluster_ci(
    values: list[float],
    groups: list[str],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    keys, means = _paper_means(values, groups)

    first = means[0]
    if bool((means == first).all()):
        # Every resample is a mean of copies of one value, so the bootstrap
        # distribution is that value. Returning it directly keeps the null
        # controls exact instead of letting float summation over thousands of
        # papers move the quantiles by an ULP.
        return (float(first), float(first))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boot = means[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))
