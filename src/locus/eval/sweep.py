"""Select lambda (and the PPMI smoothing parameters) on val.

Sweeps the pre-registered primary metric -- LOCUS-Rank MRR on the headline
set, base = SPECTER2, base vs base+rescore -- over alpha, min_count and
lambda, and reports the whole grid. Selection happens on val and only on val;
test artefacts are never read here, which is what the split suffix on every
filename is for.

Structured around what is invariant to what, because the naive triple loop
recomputes almost everything:

  base cosines     depend on nothing in the grid  -> computed once
  z-normalisation  is monotone, so also fixed     -> computed once
  PPMI means       depend on (alpha, min_count)   -> once per smoothing pair
  the sum          depends on lambda              -> the only per-lambda work

lambda = 0 is in the grid as a self-check, not for completeness: z-scoring is
strictly monotone within a pool, so MRR at lambda = 0 must equal the raw
cosine MRR *exactly*. A mismatch means the rescoring path has perturbed the
base ranking, and the sweep refuses to report a winner if it happens.

Usage:
    .venv/bin/python -m locus.eval.sweep --embeddings work/embeddings_val.parquet
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

from locus import config
from locus.core.bootstrap import paper_mean
from locus.core.probe import reciprocal_rank
from locus.core.rescore import zscore_pool
from locus.core.types import anchors_union, gold_sets
from locus.data.corpus import load_index
from locus.scoring import base_scorer
from locus.scoring.cocitation import CoGraph
from locus.scoring.dense import DenseScorer
from locus.scoring.node2vec import ASSOC_CHOICES, association

# Widened after a first pass put the optimum on both grid edges (alpha=0.5,
# lambda=3.0). A boundary optimum is not an optimum -- it is the grid running
# out, and reporting it as the selected lambda would understate the method.
ALPHAS = (0.1, 0.25, 0.5, 0.75, 1.0)
MIN_COUNTS = (1, 2)
LAMBDAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
           7.0, 10.0, 15.0, 20.0, 30.0, 50.0)


def build_pools(split: str, embeddings: Path,
                slice_: str = "headline", *, scorer=None) -> list[dict]:
    """One entry per rank item in `slice_`: base cosines, z-scores, anchors.

    headline  the pre-registered set: hard AND pool >= 10 AND |A(l)| >= 1.
    easy      the complement of hard, still requiring an anchor to rescore
              with. This exists so the claim that encoders undercapture
              location information "particularly on the hard slice" can be
              checked rather than assumed -- it is a comparative claim and
              needs the other side measured.
    all       every kept rank item, anchors or not.

    Every kept rank item already has a full pool: build_day1 drops the rest
    under `too_few_distractors`, which is what keeps the H_10/10 floor exact.
    """
    if slice_ not in ("headline", "easy", "all"):
        raise ValueError(f"unknown slice {slice_!r}")
    work = config.WORK_DIR
    with open(work / f"rank_items_{split}.pkl", "rb") as f:
        items = pickle.load(f)
    headline = set(pickle.loads((work / f"headline_ids_{split}.pkl").read_bytes()))
    hard = set(pickle.loads((work / f"hard_ids_{split}.pkl").read_bytes()))
    index = load_index(work / f"index_{split}.pkl")
    if scorer is None:
        scorer = DenseScorer(embeddings)

    anchor_of: dict[str, frozenset[str]] = {}
    for recs in index.values():
        golds = gold_sets(recs)
        for rec in recs:
            anchor_of[rec.context_id] = anchors_union(rec, golds)

    pools = []
    for item in items:
        if slice_ == "headline":
            if item.context_id not in headline:
                continue
        elif slice_ == "easy" and (
            item.context_id in hard or not anchor_of[item.context_id]
        ):
            continue
        base = scorer.pool_scores(item.context_id, item.candidates)
        pools.append({
            "context_id": item.context_id,
            "citing_id": item.citing_id,
            "gold": item.gold,
            "base": base,
            "z": zscore_pool(base),
            "anchors": anchor_of[item.context_id],
        })
    return pools


def mrr_over(pools: list[dict], score_of) -> float:
    values = [reciprocal_rank(score_of(p), p["gold"]) for p in pools]
    return paper_mean(values, [p["citing_id"] for p in pools])


def mrr_rescored(pools: list[dict], cached: list[dict], lam: float) -> float:
    """MRR of z + lam * mean PPMI, with the PPMI means already computed."""
    values = [
        reciprocal_rank(
            {c: p["z"][c] + lam * cache[c] for c in p["z"]}, p["gold"]
        )
        for p, cache in zip(pools, cached, strict=True)
    ]
    return paper_mean(values, [p["citing_id"] for p in pools])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("val",), default="val")
    ap.add_argument("--embeddings", type=Path)
    ap.add_argument("--texts", type=Path,
                    help="a (key, kind, text) parquet -- score with BM25 instead of a dense encoder; mutually exclusive with --embeddings")
    ap.add_argument("--assoc", choices=ASSOC_CHOICES, default="ppmi",
                    help="association measure. 'count' is the raw co-count "
                         "ablation and 'node2vec' the learned-embedding one; "
                         "each lives on its own scale and needs its own "
                         "lambda sweep")
    ap.add_argument("--vectors", type=Path, default=None,
                    help="node2vec prefix, for --assoc node2vec")
    ap.add_argument("--agg", choices=("mean", "masked_mean", "max"),
                    default="mean",
                    help="how the anchor set's evidence is pooled. 'mean' is "
                         "pre-registered and carries every headline number; "
                         "the others are the ablation motivated by the "
                         "negative influence of zero-PPMI anchors. Their "
                         "scales differ by roughly an order of magnitude, so "
                         "each needs its own lambda swept here")
    ap.add_argument("--graph", type=Path,
                    default=config.WORK_DIR / "cocitation_train.npz")
    ap.add_argument("--out", type=Path,
                    default=config.WORK_DIR / "sweep_val.json")
    args = ap.parse_args(argv)

    pools = build_pools(args.split, args.embeddings,
                        scorer=base_scorer(args.embeddings, args.texts))
    print(f"{len(pools)} headline pools on {args.split}")

    base_mrr = mrr_over(pools, lambda p: p["base"])
    z_mrr = mrr_over(pools, lambda p: p["z"])
    print(f"base MRR (raw cosine) : {base_mrr!r}")
    print(f"base MRR (z-scored)   : {z_mrr!r}")
    if base_mrr != z_mrr:
        raise SystemExit(
            f"z-normalisation changed the base MRR ({base_mrr!r} -> {z_mrr!r}); "
            "it is monotone within a pool, so the rescoring path is perturbing "
            "the base ranking and no lambda selected here would be meaningful"
        )

    graph = CoGraph.load(args.graph)
    grid: list[dict] = []
    # raw_counts ignores alpha, so sweeping it would run the same grid five
    # times and then report whichever arbitrary alpha won the tie.
    # Neither smoothing axis means anything outside PPMI: sweeping them would
    # run the identical grid five or ten times and then report whichever
    # arbitrary (alpha, min_count) happened to win the tie.
    alphas = ALPHAS if args.assoc == "ppmi" else (None,)
    min_counts = MIN_COUNTS if args.assoc != "node2vec" else (None,)
    for alpha in alphas:
        for min_count in min_counts:
            ppmi = association(args.assoc, graph, alpha, min_count,
                               args.vectors)
            cached = [
                {c: ppmi.aggregate(c, p["anchors"], args.agg)
                 for c in p["base"]}
                for p in pools
            ]
            nonzero = sum(
                1 for cache in cached for v in cache.values() if v > 0.0
            )
            for lam in LAMBDAS:
                mrr = mrr_rescored(pools, cached, lam)
                grid.append({
                    "alpha": alpha, "min_count": min_count, "lambda": lam,
                    "mrr": mrr, "delta": mrr - base_mrr,
                })
                if lam == 0.0 and mrr != base_mrr:
                    raise SystemExit(
                        f"lambda=0 gave {mrr!r} against a base of {base_mrr!r}; "
                        "the rescored path must reproduce the base exactly there"
                    )
            print(f"  alpha={alpha} min_count={min_count}: "
                  f"{nonzero} nonzero PPMI cells, "
                  f"best delta {max(g['delta'] for g in grid[-len(LAMBDAS):]):+.6f}")

    best = max(grid, key=lambda g: g["mrr"])
    # A boundary optimum is the grid running out, not an optimum. The lambda
    # grid was hand-widened until the mean aggregator's optimum was interior;
    # masked_mean and max sit at a different scale, so that is re-checked here
    # rather than assumed.
    if best["lambda"] in (LAMBDAS[0], LAMBDAS[-1]):
        print(f"\nWARNING: lambda={best['lambda']} is on the grid boundary "
              f"({LAMBDAS[0]}..{LAMBDAS[-1]}); widen the grid before treating "
              "this as a selected value")
    report = {
        "split": args.split,
        "assoc": args.assoc,
        "agg": args.agg,
        "pools": len(pools),
        "base_mrr": base_mrr,
        "best": best,
        "grid": grid,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nbest: alpha={best['alpha']} min_count={best['min_count']} "
          f"lambda={best['lambda']} MRR={best['mrr']!r} "
          f"(delta {best['delta']:+.6f})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
