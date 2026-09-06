"""LOCUS-Swap with a real base scorer.

The second of the two probes. Two locations in one paper with disjoint gold
sets, a forced choice between the true assignment and the swapped one, and a
provable 50% floor for any context-independent scorer.

**Scaling is global, not per pool.** A swap
pool holds two candidates, so pool z-scoring maps them to exactly +1 and -1
whatever the underlying scores were, destroying the magnitude the decision
needs. Standardisation constants are therefore computed once over a whole
split rather than per pair.

Only sigma actually matters, and that is provable rather than empirical. The
swap contrast is

    delta = (S(a,ga) + S(b,gb)) - (S(a,gb) + S(b,ga))

with two positive and two negative base terms, so under S = (base - mu)/sigma
every mu cancels exactly and delta scales by 1/sigma. Sigma is not cosmetic
either: it fixes the exchange rate between the base score and the PPMI term,
which is the whole content of lambda. `test_global_mean_cancels_exactly`
pins the first half; the lambda sweep measures the second.

Sigma is taken from **val** and applied to test. The spec asks for a train
statistic; no base scores exist on train, because only the pooled test and
val contexts were ever embedded. Val is the nearest available held-out
statistic and is already the split lambda is selected on, so the reported
split never contributes its own scaling constant.

Three arms are reported separately: base only,
cocite only, and base + lambda * cocite.

Usage:
    .venv/bin/python -m locus.eval.swap \
        --val-embeddings work/embeddings_val.parquet \
        --test-embeddings work/embeddings_test.parquet --tag specter2
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

from locus import config
from locus.core.bootstrap import cluster_ci, paper_mean
from locus.core.probe import swap_outcome
from locus.core.types import anchors_union, gold_sets
from locus.data.corpus import load_index
from locus.scoring import base_scorer
from locus.scoring.cocitation import CoGraph
from locus.scoring.dense import DenseScorer
from locus.scoring.node2vec import ASSOC_CHOICES, association

LAMBDAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
           7.0, 10.0, 15.0, 20.0, 30.0, 50.0)


def build_pairs(split: str, embeddings: Path, *, scorer=None) -> list[dict]:
    """The four base scores and two anchor sets each swap pair needs."""
    work = config.WORK_DIR
    with open(work / f"swap_pairs_{split}.pkl", "rb") as f:
        pairs = pickle.load(f)
    index = load_index(work / f"index_{split}.pkl")
    if scorer is None:
        scorer = DenseScorer(embeddings)

    anchor_of: dict[str, frozenset[str]] = {}
    for recs in index.values():
        golds = gold_sets(recs)
        for rec in recs:
            anchor_of[rec.context_id] = anchors_union(rec, golds)

    return [
        {
            "citing_id": p.citing_id,
            "gold_a": p.gold_a, "gold_b": p.gold_b,
            "anch_a": anchor_of[p.ctx_a], "anch_b": anchor_of[p.ctx_b],
            "base": (
                scorer(p.ctx_a, p.gold_a), scorer(p.ctx_b, p.gold_b),
                scorer(p.ctx_a, p.gold_b), scorer(p.ctx_b, p.gold_a),
            ),
        }
        for p in pairs
    ]


def global_stats(pairs: list[dict]) -> tuple[float, float]:
    """Mean and population sd of every base score in the split, exactly summed."""
    vals = [v for p in pairs for v in p["base"]]
    mu = math.fsum(vals) / len(vals)
    var = math.fsum((v - mu) ** 2 for v in vals) / len(vals)
    return mu, math.sqrt(var)


def _cocite(pair: dict, ppmi) -> tuple[float, float, float, float]:
    return (
        ppmi.mean_over(pair["gold_a"], pair["anch_a"]),
        ppmi.mean_over(pair["gold_b"], pair["anch_b"]),
        ppmi.mean_over(pair["gold_b"], pair["anch_a"]),
        ppmi.mean_over(pair["gold_a"], pair["anch_b"]),
    )


def accuracy(pairs, ppmi, lam, mu, sigma, arm) -> tuple[float, list, list]:
    outcomes, groups = [], []
    for p in pairs:
        if arm == "base":
            s = [(v - mu) / sigma for v in p["base"]]
        elif arm == "cocite":
            s = list(_cocite(p, ppmi))
        else:
            co = _cocite(p, ppmi)
            s = [(v - mu) / sigma + lam * c for v, c in zip(p["base"], co, strict=True)]
        outcomes.append(swap_outcome(*s))
        groups.append(p["citing_id"])
    return paper_mean(outcomes, groups), outcomes, groups


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-embeddings", type=Path)
    ap.add_argument("--test-embeddings", type=Path)
    ap.add_argument("--val-texts", type=Path,
                    help="a (key, kind, text) parquet -- score with BM25 instead of a dense encoder; mutually exclusive with --embeddings")
    ap.add_argument("--test-texts", type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sweep", type=Path, help="rank sweep, for alpha/min_count")
    ap.add_argument("--graph", type=Path,
                    default=config.WORK_DIR / "cocitation_train.npz")
    ap.add_argument("--assoc", choices=ASSOC_CHOICES, default="ppmi")
    ap.add_argument("--vectors", type=Path, default=None,
                    help="node2vec prefix, for --assoc node2vec")
    args = ap.parse_args(argv)

    sel = json.loads(args.sweep.read_text())["best"] if args.sweep else {
        "alpha": 0.5, "min_count": 1}
    alpha, min_count = sel["alpha"], sel["min_count"]
    ppmi = association(args.assoc, CoGraph.load(args.graph), alpha, min_count,
                       args.vectors)

    val = build_pairs("val", args.val_embeddings,
                      scorer=base_scorer(args.val_embeddings, args.val_texts))
    test = build_pairs("test", args.test_embeddings,
                       scorer=base_scorer(args.test_embeddings, args.test_texts))
    mu, sigma = global_stats(val)
    print(f"{len(val)} val pairs, {len(test)} test pairs")
    print(f"global standardisation from val: mu={mu!r} sigma={sigma!r}")
    print(f"association={args.assoc} alpha={alpha} min_count={min_count}")

    # lambda on val, with the val-derived constants.
    grid = []
    for lam in LAMBDAS:
        acc, _, _ = accuracy(val, ppmi, lam, mu, sigma, "both")
        grid.append({"lambda": lam, "accuracy": acc})
    best = max(grid, key=lambda g: g["accuracy"])
    print(f"selected on val: lambda={best['lambda']} (val acc {best['accuracy']:.6f})")

    report = {"tag": args.tag, "alpha": alpha, "min_count": min_count,
              "mu": mu, "sigma": sigma, "val_grid": grid,
              "selected_lambda": best["lambda"], "test": {}}
    print(f"\n{'arm':<14} {'accuracy':>10}  95% CI")
    kept = {}
    for arm, lam in (("base", 0.0), ("cocite", 0.0),
                     ("both", best["lambda"])):
        acc, outcomes, groups = accuracy(test, ppmi, lam, mu, sigma, arm)
        lo, hi = cluster_ci(outcomes, groups, n_boot=1000, seed=config.SEED)
        report["test"][arm] = {"accuracy": acc, "ci": [lo, hi], "lambda": lam}
        kept[arm] = (outcomes, groups)
        print(f"{arm:<14} {acc:>10.6f}  [{lo:.6f}, {hi:.6f}]")

    # Paired, like the rank gate: every arm scores the same pairs in the same
    # order, so the difference is taken per pair before resampling. Reading two
    # marginal intervals for overlap is a different, weaker question.
    groups = kept["base"][1]
    print(f"\n{'paired delta':<14} {'delta':>10}  95% CI")
    for name, a, b in (("both-base", "both", "base"),
                       ("both-cocite", "both", "cocite")):
        diffs = [x - y for x, y in zip(kept[a][0], kept[b][0], strict=True)]
        lo, hi = cluster_ci(diffs, groups, n_boot=1000, seed=config.SEED)
        d = paper_mean(diffs, groups)
        report["test"][name] = {"delta": d, "ci": [lo, hi],
                                "excludes_zero": lo > 0.0 or hi < 0.0}
        print(f"{name:<14} {d:>+10.6f}  [{lo:+.6f}, {hi:+.6f}]"
              f"  {'*' if lo > 0.0 or hi < 0.0 else ''}")

    out = config.WORK_DIR / f"swap_report_{args.tag}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
