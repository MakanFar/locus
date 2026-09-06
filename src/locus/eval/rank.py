"""The rank gate: does the structural signal help, and is it location-specific?

Two questions, both settled by a paired paper-level cluster bootstrap on test,
at the alpha / min_count / lambda selected on val. Passing both is the
precondition for the hybrid framing:

  GATE 1  the pre-registered primary metric -- LOCUS-Rank MRR on the headline
          set (hard AND pool >= 10 AND |A(l)| >= 1), base = SPECTER2, base vs
          base+rescore -- improves with a 95% CI excluding 0.
          Fails -> diagnosis-led framing, do not headline the hybrid.

  GATE 2  oracle same-location anchors beat SAME-PAPER OTHER-LOCATION anchors
          on the same test. This is the single most load-bearing control in
          the paper: random anchors vary topic
          and location at once, so beating them proves only that same-paper
          anchors help. Substituting A(l') from another location in the SAME
          paper holds paper-topic exactly constant and varies only location.
          Fails -> the signal is paper-level topical coherence, not location,
          and the thesis has to be rewritten before the abstract is.

The bootstrap is paired: both conditions score the same pools in the same
order, so the per-item difference is taken first and the CI is computed on
those differences. An unpaired CI on two separately-resampled means would be
far wider and would answer a question nobody asked.

Usage:
    .venv/bin/python -m locus.eval.rank \
        --embeddings work/embeddings_test.parquet --sweep work/sweep_val.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from locus import config
from locus.core.bootstrap import cluster_ci, paper_mean
from locus.core.probe import hit_at_1, reciprocal_rank
from locus.core.types import gold_sets
from locus.data.corpus import load_index
from locus.eval.sweep import build_pools
from locus.scoring import base_scorer
from locus.scoring.cocitation import CoGraph
from locus.scoring.node2vec import ASSOC_CHOICES, association


def other_location_anchors(split: str, seed: int) -> dict[str, frozenset[str]]:
    """A(l') for a seeded other location l' in the same citing paper.

    Contexts whose paper offers no second location with a non-empty gold set
    get no entry: there is no control to draw, and inventing one (an empty
    anchor set, say) would compare "other-location anchors" against "no
    anchors" and quietly answer a different question.
    """
    index = load_index(config.WORK_DIR / f"index_{split}.pkl")
    out: dict[str, frozenset[str]] = {}
    for citing_id, recs in sorted(index.items()):
        golds = gold_sets(recs)
        rng = random.Random(f"{seed}:{citing_id}")
        for rec in sorted(recs, key=lambda r: r.context_id):
            others = sorted(
                loc for loc, g in golds.items() if loc != rec.location and g
            )
            if not others:
                continue
            out[rec.context_id] = frozenset(golds[rng.choice(others)])
    return out


def _score(pool: dict, ppmi, anchors, lam: float, metric=reciprocal_rank,
           agg: str = "mean") -> float:
    """Score one pool under `metric`.

    The lam == 0 shortcut reads the raw base rather than the z-scored copy.
    That is safe for any rank-based metric because z-scoring is monotone, and
    it is the same shortcut sweep.py relies on to make lambda = 0 reproduce
    the base exactly.
    """
    if lam == 0.0:
        return metric(pool["base"], pool["gold"])
    scores = {
        c: pool["z"][c] + lam * ppmi.aggregate(c, anchors, agg)
        for c in pool["z"]
    }
    return metric(scores, pool["gold"])


def paired(a: list[float], b: list[float], groups: list[str], seed: int) -> dict:
    """Paper-level cluster bootstrap on the per-item difference a - b."""
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    lo, hi = cluster_ci(diffs, groups, n_boot=1000, seed=seed)
    return {
        "a": paper_mean(a, groups),
        "b": paper_mean(b, groups),
        "delta": paper_mean(diffs, groups),
        "ci": [lo, hi],
        "excludes_zero": lo > 0.0 or hi < 0.0,
        "n": len(diffs),
    }


COVERAGE_STRATA = ("target", "distractor", "none")


def coverage_stratum(pool: dict, ppmi, agg: str = "mean") -> str:
    """Which relation, if any, the training graph captured for this pool.

      target      PPMI(gold, A) > 0 -- the graph holds the relation the
                  rescorer needs, whatever else it also holds
      distractor  the gold is unseen but some distractor is co-cited with the
                  anchors, so the structural term can only push the gold down
      none        the graph is silent on every candidate; the rescorer is a
                  no-op here by construction

    Exhaustive and exclusive. This is the stratification behind the paper's
    coverage table: the aggregate gain is the sum of a large positive effect
    on `target`, a negative one on `distractor`, and exactly zero on `none`.
    """
    anchors = pool["anchors"]
    if ppmi.aggregate(pool["gold"], anchors, agg) > 0.0:
        return "target"
    if any(ppmi.aggregate(c, anchors, agg) > 0.0
           for c in pool["z"] if c != pool["gold"]):
        return "distractor"
    return "none"


def coverage_strata(pools: list[dict], ppmi, lam: float, agg: str,
                    seed: int) -> dict[str, dict]:
    """Paired base-vs-rescored MRR within each coverage stratum.

    Each stratum is its own paired paper-level bootstrap; citing papers may
    appear in several strata, so the rows do not recombine into the headline
    interval, only into its point estimate.
    """
    by: dict[str, list[dict]] = {s: [] for s in COVERAGE_STRATA}
    for p in pools:
        by[coverage_stratum(p, ppmi, agg)].append(p)
    # The paper's table also carries the PPMI(c, A) = 0 row -- everything the
    # graph did not capture the gold relation for -- with its own interval.
    # Reported after the partition rather than in place of its two halves,
    # because the halves are what explain its sign.
    by["unseen"] = by["distractor"] + by["none"]
    out: dict[str, dict] = {}
    for stratum, sub in by.items():
        if not sub:
            out[stratum] = {"n": 0, "share": 0.0, "a": None, "b": None,
                            "delta": None, "ci": [None, None]}
            continue
        groups = [p["citing_id"] for p in sub]
        base = [_score(p, ppmi, p["anchors"], 0.0, agg=agg) for p in sub]
        resc = [_score(p, ppmi, p["anchors"], lam, agg=agg) for p in sub]
        row = paired(resc, base, groups, seed)
        row["share"] = len(sub) / len(pools)
        out[stratum] = row
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test")
    ap.add_argument("--embeddings", type=Path)
    ap.add_argument("--texts", type=Path,
                    help="a (key, kind, text) parquet -- score with BM25 instead of a dense encoder; mutually exclusive with --embeddings")
    ap.add_argument("--sweep", type=Path,
                    default=config.WORK_DIR / "sweep_val.json")
    ap.add_argument("--assoc", choices=ASSOC_CHOICES, default="ppmi",
                    help="association measure; must match the --assoc the "
                         "supplied --sweep was run with, since lambda is not "
                         "transferable across measures")
    ap.add_argument("--vectors", type=Path, default=None,
                    help="node2vec prefix, for --assoc node2vec")
    ap.add_argument("--agg", choices=("mean", "masked_mean", "max"),
                    default="mean",
                    help="anchor-set aggregator; must match the --agg the "
                         "supplied --sweep was run with, since lambda is not "
                         "transferable across aggregators")
    ap.add_argument("--graph", type=Path,
                    default=config.WORK_DIR / "cocitation_train.npz")
    ap.add_argument("--slice", dest="slice_", default="headline",
                    choices=("headline", "easy", "all"))
    ap.add_argument("--out", type=Path,
                    default=config.WORK_DIR / "rank_report.json")
    args = ap.parse_args(argv)

    selected = json.loads(args.sweep.read_text())["best"]
    alpha, min_count, lam = (
        selected["alpha"], selected["min_count"], selected["lambda"]
    )
    print(f"selected on val: alpha={alpha} min_count={min_count} lambda={lam}")

    pools = build_pools(args.split, args.embeddings, args.slice_,
                        scorer=base_scorer(args.embeddings, args.texts))
    _graph = CoGraph.load(args.graph)
    ppmi = association(args.assoc, _graph, alpha, min_count, args.vectors)
    control = other_location_anchors(args.split, config.SEED)
    failures: list[str] = []

    # GATE 1 runs on every headline pool.
    groups = [p["citing_id"] for p in pools]
    base = [_score(p, ppmi, p["anchors"], 0.0, agg=args.agg) for p in pools]
    rescored = [_score(p, ppmi, p["anchors"], lam, agg=args.agg) for p in pools]
    gate1 = paired(rescored, base, groups, config.SEED)
    print(f"\nGATE 1  base {gate1['b']!r} -> rescored {gate1['a']!r}")
    print(f"        delta {gate1['delta']:+.6f} "
          f"CI [{gate1['ci'][0]:+.6f}, {gate1['ci'][1]:+.6f}] "
          f"{'PASS' if gate1['excludes_zero'] and gate1['delta'] > 0 else 'FAIL'}")
    if not (gate1["excludes_zero"] and gate1["delta"] > 0):
        failures.append("gate 1: rescoring does not improve the primary metric")

    # GATE 2 runs only where a control anchor set exists, and both arms are
    # restricted to that same subset so the comparison stays paired.
    sub = [p for p in pools if p["context_id"] in control]
    print(f"\nGATE 2  {len(sub)} of {len(pools)} pools have a second location")
    sub_groups = [p["citing_id"] for p in sub]
    same = [_score(p, ppmi, p["anchors"], lam, agg=args.agg) for p in sub]
    other = [_score(p, ppmi, control[p["context_id"]], lam, agg=args.agg) for p in sub]
    gate2 = paired(same, other, sub_groups, config.SEED)
    print(f"        other-location {gate2['b']!r} -> same-location {gate2['a']!r}")
    print(f"        delta {gate2['delta']:+.6f} "
          f"CI [{gate2['ci'][0]:+.6f}, {gate2['ci'][1]:+.6f}] "
          f"{'PASS' if gate2['excludes_zero'] and gate2['delta'] > 0 else 'FAIL'}")
    if not (gate2["excludes_zero"] and gate2["delta"] > 0):
        failures.append(
            "gate 2: same-location anchors do not beat same-paper "
            "other-location anchors -- the signal is paper-level topical "
            "coherence, not location"
        )
    # The same control read against the BASE rather than against the
    # same-location arm: this is the row the paper's controls table prints
    # (base -> other-location rescored), and gate 2's interval does not cover
    # it because gate 2 pairs other against same.
    base_sub = [_score(p, ppmi, p["anchors"], 0.0, agg=args.agg) for p in sub]
    other_vs_base = paired(other, base_sub, sub_groups, config.SEED)
    print(f"        vs base: {other_vs_base['b']!r} -> {other_vs_base['a']!r}"
          f"  delta {other_vs_base['delta']:+.6f} "
          f"CI [{other_vs_base['ci'][0]:+.6f}, {other_vs_base['ci'][1]:+.6f}]")

    # R@1 is reported alongside MRR but never gates anything: the
    # pre-registered primary is MRR, and adding a second metric to the gate
    # would reopen the multiple-comparisons hole section 6 closed. Both use
    # the expectation over random tie-breaking (probe.py), so they are on the
    # same convention and can sit in one table.
    base1 = [_score(p, ppmi, p["anchors"], 0.0, hit_at_1, args.agg) for p in pools]
    resc1 = [_score(p, ppmi, p["anchors"], lam, hit_at_1, args.agg) for p in pools]
    gate1_r1 = paired(resc1, base1, groups, config.SEED)
    same1 = [_score(p, ppmi, p["anchors"], lam, hit_at_1, args.agg) for p in sub]
    other1 = [_score(p, ppmi, control[p["context_id"]], lam, hit_at_1, args.agg)
              for p in sub]
    gate2_r1 = paired(same1, other1, sub_groups, config.SEED)
    print(f"\nR@1     base {gate1_r1['b']:.6f} -> rescored {gate1_r1['a']:.6f}"
          f"  delta {gate1_r1['delta']:+.6f} "
          f"CI [{gate1_r1['ci'][0]:+.6f}, {gate1_r1['ci'][1]:+.6f}]")
    print(f"        other-location {gate2_r1['b']:.6f} -> "
          f"same-location {gate2_r1['a']:.6f}"
          f"  delta {gate2_r1['delta']:+.6f} "
          f"CI [{gate2_r1['ci'][0]:+.6f}, {gate2_r1['ci'][1]:+.6f}]")

    # The coverage table: the headline delta decomposed by what the training
    # graph knew about each pool. Not a gate -- a reading of the same numbers.
    strata = coverage_strata(pools, ppmi, lam, args.agg, config.SEED)
    print(f"\n{'coverage':<12} {'pools':>7} {'share':>7} {'base':>9} "
          f"{'+resc':>9} {'delta':>10}  95% CI")
    for name, row in strata.items():
        if not row["n"]:
            print(f"{name:<12} {0:>7}")
            continue
        print(f"{name:<12} {row['n']:>7} {row['share']:>7.3f} {row['b']:>9.5f} "
              f"{row['a']:>9.5f} {row['delta']:>+10.5f}  "
              f"[{row['ci'][0]:+.5f}, {row['ci'][1]:+.5f}]")

    report = {
        "split": args.split,
        "slice": args.slice_,
        "assoc": args.assoc,
        "agg": args.agg,
        "selected": selected,
        "pools": len(pools),
        "gate1_primary": gate1,
        "gate2_other_location_control": gate2,
        "other_location_vs_base": other_vs_base,
        "gate1_hit_at_1": gate1_r1,
        "gate2_hit_at_1": gate2_r1,
        "coverage_strata": strata,
        "failures": failures,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")

    if failures:
        print("\nGATE FAILED:")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    print("\nBoth rank gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
