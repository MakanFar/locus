# tests/equivalence/golden.py
"""A deterministic digest of the numeric primitives the benchmark reports.

The point is coverage of the *arithmetic*, not of the pipeline. Floats are
recorded with `repr`, which round-trips exactly, so the comparison is bit-exact
rather than decimal-exact.

What the digest actually covers is exactly what this module imports and calls:

    data.export        load_rank, load_swap, load_target_counts
    scoring.baselines  constant_scorer, popularity_scorer, rank_mrr,
                       swap_accuracy -- and through them core.probe's
                       reciprocal_rank and swap_outcome, but only on the
                       all-tied and integer-count pools those nulls produce
    scoring.cocitation CoGraph.load, .ppmi, .value, .mean_over
    core.bootstrap     paper_mean, cluster_ci
    core.rescore       zscore_pool, rescore

Everything else is outside it. In particular the digest never constructs a
`DenseScorer`, so a perturbation in `scoring.dense` moves no key here; nor does
it touch `core.pools`, `core.hardness`, `core.alignment`, `scoring.retrieve`,
`scoring.node2vec`, `build/` or `eval/`. The headline number
that runs through `scoring.dense`, `core.probe` and `eval.verify` is covered by
`test_headline.py` instead, and `config.POOL_SIZE` by `test_frozen_config.py`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from locus.core.bootstrap import cluster_ci, paper_mean
from locus.core.rescore import rescore, zscore_pool
from locus.data.export import load_rank, load_swap, load_target_counts
from locus.scoring.baselines import (
    constant_scorer,
    popularity_scorer,
    rank_mrr,
    swap_accuracy,
)
from locus.scoring.cocitation import CoGraph

SPLIT = "test"


def digest(work: Path) -> dict[str, str]:
    """Every value here is load-bearing in the paper or in a gate."""
    out: dict[str, str] = {}

    rank_items = load_rank(work / "export" / f"locus_rank_{SPLIT}.jsonl")
    swap_pairs = load_swap(work / "export" / f"locus_swap_{SPLIT}.jsonl")

    out["n_rank_items"] = repr(len(rank_items))
    out["n_swap_pairs"] = repr(len(swap_pairs))

    # Null controls: the exact-value assertions the benchmark rests on.
    mrr, values, groups = rank_mrr(rank_items, constant_scorer(1.0))
    lo, hi = cluster_ci(values, groups, n_boot=1000, seed=0)
    out["rank_constant_mrr"] = repr(mrr)
    out["rank_constant_ci"] = repr((lo, hi))
    out["rank_constant_distinct_values"] = repr(sorted(set(values)))

    # Counts come from the released anchors sidecar, not recomputed from
    # rank_items: RankItem.gold is a single id, not a set, so counting over
    # it would iterate characters and silently degenerate to zero matches.
    # The sidecar is index-wide scope -- the same scope the real gate uses
    # -- so this key also cross-checks the published rank_popularity number
    # in build_report_test.json.
    counts = load_target_counts(work / "export" / f"locus_anchors_{SPLIT}.jsonl")
    pop_mrr, pop_values, pop_groups = rank_mrr(rank_items, popularity_scorer(counts))
    pop_ci = cluster_ci(pop_values, pop_groups, n_boot=1000, seed=0)
    out["rank_popularity_mrr"] = repr(pop_mrr)
    out["rank_popularity_ci"] = repr(pop_ci)

    acc, outcomes, sgroups = swap_accuracy(swap_pairs, constant_scorer(1.0))
    out["swap_constant_acc"] = repr(acc)
    out["swap_constant_ci"] = repr(cluster_ci(outcomes, sgroups, n_boot=1000, seed=0))

    # Aggregation primitives, exercised on a deterministic synthetic input so
    # the digest still moves if the artefacts are ever rebuilt.
    xs = [1.0 / i for i in range(1, 1001)]
    gs = [f"p{i % 37}" for i in range(1, 1001)]
    out["paper_mean_synthetic"] = repr(paper_mean(xs, gs))
    out["cluster_ci_synthetic"] = repr(cluster_ci(xs, gs, n_boot=200, seed=0))

    # Standardisation and the rescorer.
    pool = {f"c{i}": math.sin(i) for i in range(10)}
    out["zscore_pool"] = repr(sorted(zscore_pool(pool).items()))

    graph = CoGraph.load(work / "cocitation_train.npz")
    ppmi = graph.ppmi(alpha=0.5, min_count=1)
    marg = np.asarray(graph.counts.sum(axis=1)).ravel()
    inverse = {v: k for k, v in graph.index.items()}
    # Descending marginal count, stable, so ties resolve by the graph's own
    # index order rather than by sort implementation. Lexicographic ids put
    # the 64 LEAST-connected papers in the sample, whose mutual PPMI is all
    # zero.
    vocab = [inverse[i] for i in np.argsort(-marg, kind="stable")[:64]]
    anchors = frozenset(vocab[:4])
    ppmi_values = [ppmi.value(c, a) for c in vocab[:16] for a in sorted(anchors)]
    ppmi_mean_over = [ppmi.mean_over(c, anchors) for c in vocab[:16]]
    out["ppmi_values"] = repr(ppmi_values)
    out["ppmi_mean_over"] = repr(ppmi_mean_over)
    base = {c: math.cos(i) for i, c in enumerate(vocab[:10])}
    base_z = zscore_pool(base)
    rescored = rescore(base, anchors, ppmi, 0.2)
    out["rescore"] = repr(sorted(rescored.items()))

    # Coverage guards: Findings 1 and 2 (review round 1) both returned
    # plausible-looking numbers instead of failing outright, which is why a
    # zero-coverage vocab slice and a degenerate popularity counter both
    # survived undetected. These assertions make that class of bug fail
    # loudly instead of silently blessing a digest that measures nothing.
    n_nonzero_values = sum(1 for v in ppmi_values if v != 0.0)
    if n_nonzero_values < 32:
        raise ValueError(
            f"ppmi_values coverage collapsed: {n_nonzero_values}/64 non-zero, "
            "need >= 32 -- the sampled vocab has too little mutual PPMI"
        )
    if any(v == 0.0 for v in ppmi_mean_over):
        raise ValueError(
            "ppmi_mean_over coverage collapsed: at least one of 16 entries "
            "is exactly 0.0 -- an anchor/candidate pair never co-occurs"
        )
    if sorted(rescored.items()) == sorted(base_z.items()):
        raise ValueError(
            "rescore equals zscore_pool(base): the PPMI term contributed "
            "nothing, so this key duplicates the zscore_pool key"
        )
    if pop_mrr == mrr:
        raise ValueError(
            "rank_popularity_mrr equals rank_constant_mrr: the popularity "
            "scorer degenerated to the constant scorer"
        )

    return out


def write_golden(path: Path, values: dict[str, str]) -> None:
    text = json.dumps(values, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def load_golden(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))
