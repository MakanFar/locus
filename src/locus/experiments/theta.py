"""Sensitivity of the headline result to the location-clustering threshold.

Locations are inferred by Jaccard clustering at theta = 0.5, with no ground
truth. This module asks how much the published result depends on that choice,
and it is careful about what a comparison across thresholds can mean, because
changing theta changes three things at once:

  the population   which contexts have an anchor and a full pool
  the pools        distractors are sampled from references OUTSIDE the
                   location's gold set, and that set moves with theta -- so
                   the same context usually gets a different nine distractors
  the anchors      the thing the analysis is actually about

Four readings are therefore reported for every threshold:

  full rebuild       everything rebuilt at that theta: index, pools, hardness,
                     the train-only graph, and lambda/alpha re-selected on a
                     validation split rebuilt at the same theta
  fixed cohort       the same, restricted to contexts that are in the headline
                     set at EVERY threshold
  unchanged pools    further restricted to contexts whose ten candidates are
                     identical at every threshold. A pool's distractors are
                     sampled from outside the location's cited set, so a pool
                     survives only where that set does -- and there the
                     anchors have not moved either. This reading therefore
                     isolates the TRAIN side: the graph and its lambda
  swapped anchors    the reference pools (theta = 0.5) kept exactly as they
                     are and rescored with the anchors each threshold infers,
                     against the reference graph and lambda. This is the
                     reading that isolates the anchors on the test side

**This is a sensitivity analysis, not a way to choose theta.** The benchmark
was frozen at 0.5 before any test scoring; the numbers here are computed on
test and must not be used to move it.

Each threshold lives in its own work directory, produced by

    LOCUS_WORK_DIR=work/theta_0.3 locus build pipeline --split test --theta 0.3
    LOCUS_WORK_DIR=work/theta_0.3 locus build pipeline --split val  --theta 0.3
    LOCUS_WORK_DIR=work/theta_0.3 locus build graph --theta 0.3
    LOCUS_WORK_DIR=work/theta_0.3 locus sweep --embeddings <val vectors>

Usage:
    .venv/bin/python -m locus.experiments.theta \\
        --dir 0.3=work/theta_0.3 --dir 0.5=work --dir 0.7=work/theta_0.7
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

from locus import config
from locus.core.probe import reciprocal_rank
from locus.core.rescore import rescore
from locus.core.types import anchors_union, gold_sets
from locus.data.corpus import load_index
from locus.eval.rank import paired, pool_level
from locus.eval.sweep import build_pools
from locus.scoring.cocitation import CoGraph

REFERENCE = 0.5


def cohorts(per_theta: dict[float, dict[str, dict]]) -> tuple[set[str], set[str]]:
    """(contexts eligible at every theta, those whose pool never changes)."""
    thetas = sorted(per_theta)
    common = set(per_theta[thetas[0]])
    for t in thetas[1:]:
        common &= set(per_theta[t])
    same_pool = {
        c for c in common
        if len({frozenset(per_theta[t][c]["candidates"]) for t in thetas}) == 1
    }
    return common, same_pool


def swap_cohort(reference: dict[str, dict],
                anchors_by_theta: dict[float, dict[str, frozenset[str]]]) -> set[str]:
    """Reference pools that are a clean test of every threshold's anchors.

    A pool qualifies when, at every threshold, its context has at least one
    anchor and none of them is among its ten candidates. The second condition
    matters below the reference threshold: a coarser clustering can turn one
    of the pool's distractors into an anchor, and a candidate the system is
    told is already cited at the location is no longer a distractor.
    """
    keep = set()
    for c, row in reference.items():
        cands = set(row["candidates"])
        if all(a.get(c) and not (a[c] & cands) for a in anchors_by_theta.values()):
            keep.add(c)
    return keep


def summarise_swapped(reference: dict[str, dict], anchors: dict[str, frozenset[str]],
                      ppmi, lam: float, keep: set[str], seed: int) -> dict:
    """The reference pools in `keep`, rescored with another anchor set."""
    ids = sorted(c for c in reference if c in keep)
    if not ids:
        return {"n": 0}
    groups = [reference[c]["citing_id"] for c in ids]
    base, resc = [], []
    for c in ids:
        row = reference[c]
        base.append(reciprocal_rank(row["base_scores"], row["gold"]))
        resc.append(reciprocal_rank(
            rescore(row["base_scores"], anchors[c], ppmi, lam), row["gold"]))
    out = paired(resc, base, groups, seed)
    pl = pool_level(resc, base, groups, seed)
    out["pool_a"], out["pool_b"], out["pool_delta"] = pl["a"], pl["b"], pl["delta"]
    out["pool_ci"] = pl["ci"]
    out["mean_anchors"] = sum(len(anchors[c]) for c in ids) / len(ids)
    return out


def anchors_in(work: Path, split: str) -> dict[str, frozenset[str]]:
    """context_id -> union-route anchors, from one threshold's frozen index."""
    out = {}
    for recs in load_index(work / f"index_{split}.pkl").values():
        golds = gold_sets(recs)
        for rec in recs:
            out[rec.context_id] = frozenset(anchors_union(rec, golds))
    return out


def summarise(rows: dict[str, dict], keep: set[str] | None, seed: int) -> dict:
    """Base vs rescored over `rows`, optionally restricted to a cohort."""
    ids = sorted(rows) if keep is None else sorted(c for c in rows if c in keep)
    if not ids:
        return {"n": 0}
    groups = [rows[c]["citing_id"] for c in ids]
    base = [rows[c]["base"] for c in ids]
    resc = [rows[c]["rescored"] for c in ids]
    out = paired(resc, base, groups, seed)
    pl = pool_level(resc, base, groups, seed)
    out["pool_a"], out["pool_b"], out["pool_delta"] = pl["a"], pl["b"], pl["delta"]
    out["pool_ci"] = pl["ci"]
    out["mean_anchors"] = sum(rows[c]["n_anchors"] for c in ids) / len(ids)
    return out


def score_dir(work: Path, embeddings: Path, graph: Path, sweep: Path,
              split: str = "test") -> tuple[dict[str, dict], dict]:
    """Per-context reciprocal ranks for one threshold's rebuilt artefacts."""
    previous = config.WORK_DIR
    config.WORK_DIR = work
    try:
        pools = build_pools(split, embeddings)
        with open(work / f"rank_items_{split}.pkl", "rb") as f:
            candidates = {it.context_id: it.candidates for it in pickle.load(f)}
    finally:
        config.WORK_DIR = previous
    sel = json.loads(sweep.read_text())["best"]
    ppmi = CoGraph.load(graph).ppmi(alpha=sel["alpha"], min_count=sel["min_count"])
    rows = {}
    for p in pools:
        scores = rescore(p["base"], p["anchors"], ppmi, sel["lambda"])
        rows[p["context_id"]] = {
            "citing_id": p["citing_id"],
            "candidates": candidates[p["context_id"]],
            "anchors": p["anchors"],
            "n_anchors": len(p["anchors"]),
            "base": reciprocal_rank(p["base"], p["gold"]),
            "rescored": reciprocal_rank(scores, p["gold"]),
            "base_scores": p["base"],
            "gold": p["gold"],
        }
    return rows, sel


def _composition(work: Path, split: str) -> dict:
    r = json.loads((work / f"build_report_{split}.json").read_text())
    n = r["contexts"]
    return {
        "theta": r["config"]["theta"],
        "locations": r["num_locations"],
        "anchored_clustering": r["anchor_ge1_clustering"] / n,
        "anchored_marker": r["anchor_ge1_marker"] / n,
        "anchored_union": r["anchor_ge1_union"] / n,
        "marker_outside_cluster": r["clustering_disagreement"] / max(r["marker_anchors_total"], 1),
        "rank_pools": r["rank_items"],
        "headline_pools": r["headline"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", action="append", required=True, metavar="THETA=PATH",
                    help="a work directory rebuilt at THETA; repeat per threshold")
    ap.add_argument("--embeddings-name", default="embeddings_test.parquet",
                    help="file name of the test vectors inside each directory")
    ap.add_argument("--fixed-graph", type=Path, default=None,
                    help="also score every threshold against this graph with "
                         "--fixed-sweep's parameters (isolates test-side inference)")
    ap.add_argument("--fixed-sweep", type=Path, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=config.WORK_DIR / "theta_sensitivity.json")
    args = ap.parse_args(argv)

    dirs = {}
    for spec in args.dir:
        theta, _, path = spec.partition("=")
        dirs[float(theta)] = Path(path)
    if REFERENCE not in dirs:
        raise SystemExit(f"the reference threshold {REFERENCE} must be among --dir")

    per_theta, selected, fixed = {}, {}, {}
    for theta, work in sorted(dirs.items()):
        emb = work / args.embeddings_name
        per_theta[theta], selected[theta] = score_dir(
            work, emb, work / "cocitation_train.npz", work / "sweep_val.json", args.split)
        if args.fixed_graph is not None:
            fixed[theta], _ = score_dir(work, emb, args.fixed_graph, args.fixed_sweep, args.split)
        print(f"theta {theta}: {len(per_theta[theta])} headline pools, "
              f"selected {selected[theta]}", flush=True)

    common, same_pool = cohorts(per_theta)
    ref = per_theta[REFERENCE]
    ref_sel = selected[REFERENCE]
    ref_ppmi = CoGraph.load(dirs[REFERENCE] / "cocitation_train.npz").ppmi(
        alpha=ref_sel["alpha"], min_count=ref_sel["min_count"])
    anchors_by_theta = {t: anchors_in(dirs[t], args.split) for t in sorted(dirs)}
    swap = swap_cohort(ref, anchors_by_theta)
    anchors_fixed = sum(
        len({anchors_by_theta[t].get(c) for t in dirs}) == 1 for c in same_pool)
    report = {"split": args.split, "reference": REFERENCE, "averaging": "paper",
              "cohort_contexts": len(common), "fixed_pool_contexts": len(same_pool),
              "fixed_pool_contexts_with_identical_anchors": anchors_fixed,
              "swap_cohort_contexts": len(swap),
              "thetas": {}}
    for theta in sorted(dirs):
        rows = per_theta[theta]
        shared = [c for c in rows if c in ref]
        entry = {
            "composition": _composition(dirs[theta], args.split),
            "selected": selected[theta],
            "in_reference_set": len(shared) / len(ref),
            "same_pool_as_reference": sum(
                rows[c]["candidates"] == ref[c]["candidates"] for c in shared) / len(shared),
            "same_anchors_as_reference": sum(
                rows[c]["anchors"] == ref[c]["anchors"] for c in shared) / len(shared),
            "full_rebuild": summarise(rows, None, config.SEED),
            "fixed_cohort": summarise(rows, common, config.SEED),
            "fixed_pools": summarise(rows, same_pool, config.SEED),
            "swapped_anchors": summarise_swapped(
                ref, anchors_by_theta[theta], ref_ppmi, ref_sel["lambda"], swap, config.SEED),
            "swapped_anchors_changed": sum(
                anchors_by_theta[theta][c] != anchors_by_theta[REFERENCE][c] for c in swap
            ) / max(len(swap), 1),
        }
        if fixed:
            entry["fixed_graph_full"] = summarise(fixed[theta], None, config.SEED)
            entry["fixed_graph_fixed_pools"] = summarise(fixed[theta], same_pool, config.SEED)
        report["thetas"][str(theta)] = entry

    print(f"\ncohort: {len(common)} contexts eligible at every theta; "
          f"{len(same_pool)} with an identical pool throughout "
          f"({anchors_fixed} of them with identical anchors too); "
          f"{len(swap)} reference pools usable with every theta's anchors")
    cols = ["full_rebuild", "fixed_cohort", "fixed_pools", "swapped_anchors"] + (
        ["fixed_graph_full", "fixed_graph_fixed_pools"] if fixed else [])
    print(f"{'theta':>6}{'pools':>8}{'lam':>6}{'alpha':>6}" + "".join(f"{c:>26}" for c in cols))
    for theta in sorted(dirs):
        e = report["thetas"][str(theta)]
        cells = "".join(
            f"{e[c]['delta']:>+10.5f} [{e[c]['ci'][0]:+.4f},{e[c]['ci'][1]:+.4f}]" for c in cols)
        print(f"{theta:>6}{e['full_rebuild']['n']:>8}{e['selected']['lambda']:>6}"
              f"{e['selected']['alpha']:>6}{cells}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
