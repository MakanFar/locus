"""Context-independent scorers -- the harness unit test.

By the lemma, any scorer that ignores the query context is pinned to exactly
50% on LOCUS-Swap regardless of how good its document scoring is. These are
not baselines to be beaten; they are assertions about the harness. A number
other than 0.5 means pool construction or score plumbing is wrong.

`constant_scorer` gives the matching assertion on LOCUS-Rank: every candidate
ties, so MRR is exactly H_10/10. `popularity_scorer` does NOT -- it varies
across candidates, so on Rank it is a weak real baseline rather than an
identity, which is what makes it a useful guard that the Rank floor is the tie
rule and not a stuck value.
"""
import math

from locus.core.bootstrap import paper_mean
from locus.core.probe import reciprocal_rank, swap_outcome
from locus.core.protocols import Scorer


def popularity_scorer(target_counts: dict[str, int]) -> Scorer:
    def score(context_id: str, candidate_id: str) -> float:
        return math.log1p(target_counts.get(candidate_id, 0))

    return score


def graph_embedding_scorer(vectors, citing_of: dict[str, str]) -> Scorer:
    """Score a candidate by its similarity to the CITING paper's node vector.

    This is the strongest query a graph embedding can pose on LOCUS, and it is
    the reason a graph embedding is a control here rather than a base. A
    citation context is not a node, so the nearest available query is the
    paper doing the citing -- which is the same for every location in it.

    The two probes then say different things about the same scorer, and the
    difference is the point:

      Swap  is lemma-pinned to exactly 0.5. The scorer cannot vary with the
            query context, the contrast is antisymmetric under exchanging the
            two locations, and the two sums are the same floats added in the
            opposite order.
      Rank  is NOT pinned. The scorer does vary across candidates, so like the
            popularity control it discriminates within a pool and lands off
            H_k/k. What it cannot do is tell one location from another.

    So a graph embedding can say which papers this paper is likely to cite,
    and cannot say where.
    """
    def score(context_id: str, candidate_id: str) -> float:
        citing = citing_of.get(context_id)
        if citing is None:
            return 0.0
        return vectors.value(citing, candidate_id)

    return score


def constant_scorer(value: float = 0.0) -> Scorer:
    def score(context_id: str, candidate_id: str) -> float:
        return value

    return score


def swap_accuracy(pairs, scorer: Scorer) -> tuple[float, list[float], list[str]]:
    outcomes, groups = [], []
    for p in pairs:
        outcomes.append(
            swap_outcome(
                scorer(p.ctx_a, p.gold_a),
                scorer(p.ctx_b, p.gold_b),
                scorer(p.ctx_a, p.gold_b),
                scorer(p.ctx_b, p.gold_a),
            )
        )
        groups.append(p.citing_id)
    # Paper-weighted, to match cluster_ci: the point estimate is the mean over papers.
    return paper_mean(outcomes, groups), outcomes, groups


def rank_mrr(items, scorer: Scorer) -> tuple[float, list[float], list[str]]:
    values, groups = [], []
    for it in items:
        scores = {c: scorer(it.context_id, c) for c in it.candidates}
        values.append(reciprocal_rank(scores, it.gold))
        groups.append(it.citing_id)
    return paper_mean(values, groups), values, groups
