"""The training-free co-citation rescorer.

    S(c|l) = S_base_hat(c|l) + lambda * (1/|A(l)|) * sum_{a in A(l)} PPMI(c,a)

Base scores are z-normalised within each pool. That is what the hat in Eq. 1
denotes, and it is what lets one lambda transfer across base retrievers --
which the paper claims. lambda = 0 therefore reproduces the base ranking
exactly, because z-normalisation is monotone.

Three degenerate cases must never produce NaN, because probe._band raises on
any NaN in a rank pool -- but only one of them is guarded here. A
zero-variance base pool is guarded in this module's zscore_pool, which
returns all zeros rather than dividing by a zero standard deviation. An empty
anchor set is guarded in PPMI.mean_over (cocitation.py), which returns 0.0
for anchors=frozenset() rather than dividing by zero. A candidate that is
itself one of the anchors is safe because count_sites (cocitation.py) never
records self-pairs, so PPMI(c, c) is always 0 -- an invariant of that module,
not a guard implemented here.
"""
import math

from locus.core.protocols import Association


def zscore_pool(scores: dict[str, float]) -> dict[str, float]:
    """Standardise a pool's base scores. All-equal input gives all zeros."""
    vals = list(scores.values())
    mu = math.fsum(vals) / len(vals)
    var = math.fsum((v - mu) ** 2 for v in vals) / len(vals)
    if var == 0.0:
        return dict.fromkeys(scores, 0.0)
    sd = math.sqrt(var)
    return {k: (v - mu) / sd for k, v in scores.items()}


def rescore(
    base: dict[str, float],
    anchors: frozenset[str],
    ppmi: Association,
    lam: float,
) -> dict[str, float]:
    z = zscore_pool(base)
    return {c: z[c] + lam * ppmi.mean_over(c, anchors) for c in base}
