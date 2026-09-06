"""Anchor controls: random, and degree-matched random.

The structural claim has to beat the random-anchor control, and degree-matched
random is the sharper form of it. They answer different objections:

  random            anchors drawn uniformly from the graph vocabulary. Beating
                    this shows the rescorer is using WHICH papers are cited
                    here, not merely that some PPMI mass got added.
  degree-matched    each true anchor is replaced by a random paper of similar
                    marginal count. Beating this shows the gain is not the
                    popularity of the anchors -- a high-degree anchor has
                    nonzero PPMI with far more candidates, so uniform random
                    anchors are systematically LOW degree and are therefore a
                    weaker control than they look.

Neither is the strongest control in the paper. That is same-paper
other-location anchors (eval/rank.py, gate 2), which holds paper topic constant and
varies only location; these two vary topic and location together.

Degree matching bins the vocabulary by log marginal count and draws from the
bin the true anchor falls in, so a replacement is comparable in how connected
it is without being the same paper.

Usage:
    .venv/bin/python -m locus.eval.controls \
        --embeddings work/embeddings_test.parquet --sweep work/sweep_val.json
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import math
import random
import sys
from pathlib import Path

from locus import config
from locus.core.bootstrap import cluster_ci, paper_mean
from locus.core.probe import reciprocal_rank
from locus.eval.sweep import build_pools
from locus.scoring import base_scorer
from locus.scoring.cocitation import CoGraph


def degree_bins(graph: CoGraph) -> tuple[dict[str, int], list[list[str]], list[int]]:
    """Bin the vocabulary by floor(log2(marginal)); return lookup and members."""
    bins: dict[int, list[str]] = collections.defaultdict(list)
    of_paper: dict[str, int] = {}
    for word, count in zip(graph.vocab, graph.marginals, strict=True):
        b = math.floor(math.log2(max(int(count), 1)))
        bins[b].append(word)
        of_paper[word] = b
    return of_paper, [bins[k] for k in sorted(bins)], sorted(bins)


def random_anchors(pools, vocab: list[str], seed: int) -> list[frozenset[str]]:
    """Same-sized anchor sets drawn uniformly from the graph vocabulary."""
    out = []
    for pool in pools:
        rng = random.Random(f"{seed}:rand:{pool['context_id']}")
        n = len(pool["anchors"])
        out.append(frozenset(rng.sample(vocab, n)) if n else frozenset())
    return out


def degree_matched_anchors(pools, graph: CoGraph, seed: int) -> list[frozenset[str]]:
    """Replace each anchor with a random paper from its own degree bin."""
    of_paper, members, keys = degree_bins(graph)
    out = []
    for pool in pools:
        rng = random.Random(f"{seed}:deg:{pool['context_id']}")
        picked = set()
        for anchor in sorted(pool["anchors"]):
            b = of_paper.get(anchor)
            # Not in the graph at all: fall back to the lowest bin, which is
            # where an unseen paper's degree of 0 would place it.
            bucket = (
                members[0] if b is None else members[bisect.bisect_left(keys, b)]
            )
            picked.add(bucket[rng.randrange(len(bucket))])
        out.append(frozenset(picked))
    return out


def mrr_with(pools, anchor_sets, ppmi, lam: float) -> tuple[float, list, list]:
    values, groups = [], []
    for pool, anchors in zip(pools, anchor_sets, strict=True):
        scores = {c: pool["z"][c] + lam * ppmi.mean_over(c, anchors) for c in pool["z"]}
        values.append(reciprocal_rank(scores, pool["gold"]))
        groups.append(pool["citing_id"])
    return paper_mean(values, groups), values, groups


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test")
    ap.add_argument("--embeddings", type=Path)
    ap.add_argument("--texts", type=Path,
                    help="a (key, kind, text) parquet -- score with BM25 instead of a dense encoder; mutually exclusive with --embeddings")
    ap.add_argument("--sweep", type=Path, default=config.WORK_DIR / "sweep_val.json")
    ap.add_argument("--graph", type=Path,
                    default=config.WORK_DIR / "cocitation_train.npz")
    ap.add_argument("--tag", default="specter2")
    args = ap.parse_args(argv)

    sel = json.loads(args.sweep.read_text())["best"]
    alpha, min_count, lam = sel["alpha"], sel["min_count"], sel["lambda"]
    graph = CoGraph.load(args.graph)
    ppmi = graph.ppmi(alpha=alpha, min_count=min_count)
    pools = build_pools(args.split, args.embeddings,
                        scorer=base_scorer(args.embeddings, args.texts))
    print(f"{len(pools)} pools | alpha={alpha} min_count={min_count} lambda={lam}")

    arms = {
        "true": [p["anchors"] for p in pools],
        "random": random_anchors(pools, list(graph.vocab), config.SEED),
        "degree_matched": degree_matched_anchors(pools, graph, config.SEED),
    }
    # The base, through the same scoring path with no anchors: mean_over of an
    # empty set is exactly 0.0, and z-scoring is monotone, so this IS the base
    # ranking. Every arm is paired against it below, because the paper's
    # controls table states each arm as base -> arm, not as true -> arm.
    base_mrr, base_values, groups = mrr_with(
        pools, [frozenset()] * len(pools), ppmi, lam)
    got, mrrs = {}, {}
    print(f"\n{'arm':<16} {'MRR':>10}   vs base (95% CI)")
    print(f"{'base':<16} {base_mrr:>10.6f}")

    def _vs_base(values):
        diffs = [x - y for x, y in zip(values, base_values, strict=True)]
        lo, hi = cluster_ci(diffs, groups, n_boot=1000, seed=config.SEED)
        return {"delta": paper_mean(diffs, groups), "ci": [lo, hi]}

    for name, sets in arms.items():
        mrr, values, _ = mrr_with(pools, sets, ppmi, lam)
        got[name], mrrs[name] = values, mrr
        vb = _vs_base(values)
        print(f"{name:<16} {mrr:>10.6f}   {vb['delta']:+.6f} "
              f"[{vb['ci'][0]:+.6f}, {vb['ci'][1]:+.6f}]")

    report = {"tag": args.tag, "split": args.split, "selected": sel,
              "pools": len(pools), "base_mrr": base_mrr,
              "true_mrr": mrrs["true"], "true_vs_base": _vs_base(got["true"]),
              "arms": {}}
    print(f"\n{'paired delta':<24} {'delta':>10}  95% CI")
    for other in ("random", "degree_matched"):
        diffs = [x - y for x, y in zip(got["true"], got[other], strict=True)]
        lo, hi = cluster_ci(diffs, groups, n_boot=1000, seed=config.SEED)
        d = paper_mean(diffs, groups)
        ok = lo > 0.0
        report["arms"][other] = {"mrr": mrrs[other], "vs_base": _vs_base(got[other]),
                                 "delta": d, "ci": [lo, hi], "beats": ok}
        print(f"{'true - ' + other:<24} {d:>+10.6f}  [{lo:+.6f}, {hi:+.6f}]"
              f"  {'PASS' if ok else 'FAIL'}")

    out = config.WORK_DIR / f"controls_{args.tag}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")
    return 0 if all(v["beats"] for v in report["arms"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
