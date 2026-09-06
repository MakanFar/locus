"""Reproduce the paper's primary result from the exported files alone.

This is the acceptance test for the export: it uses `locus_rank_{split}.jsonl`
and `locus_anchors_{split}.jsonl` and never opens `index_{split}.pkl`,
`rank_items_{split}.pkl` or `headline_ids_{split}.pkl`. If the export is
missing anything a consumer needs, this is where it shows up -- as an
ImportError, a KeyError, or a number that does not match.

It doubles as the worked example for someone who has the release and wants to
score their own encoder: swap `--embeddings` for their own parquet of
`(key, kind, embedding)` and the rest is unchanged.

Two inputs are NOT in the export and cannot be:

  --embeddings    a model's vectors, which are the thing being evaluated
  --graph         the train-split co-citation counts (`cocitation_train.npz`),
                  which are ours to publish and are shipped alongside

Expected on test with SPECTER2 at alpha=0.5, min_count=1, lambda=4.0:

    MRR  0.47771 -> 0.59585   (delta +0.11814)
    R@1  0.28051 -> 0.43124   (delta +0.15072)

Usage:
    .venv/bin/python -m locus.eval.verify \\
        --export work/export --embeddings work/embeddings_test.parquet \\
        --graph work/cocitation_train.npz
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pickle
import sys
from pathlib import Path

from locus import config
from locus.core.bootstrap import cluster_ci, paper_mean
from locus.core.probe import hit_at_1, reciprocal_rank
from locus.core.rescore import rescore
from locus.core.types import anchors_union, gold_sets
from locus.data.corpus import load_index
from locus.data.export import (
    SCHEMA,
    load_anchors,
    load_rank,
    load_swap,
    load_target_counts,
    pool_size_of,
    rank_floor,
)
from locus.scoring.baselines import (
    constant_scorer,
    popularity_scorer,
    rank_mrr,
    swap_accuracy,
)
from locus.scoring.cocitation import CoGraph
from locus.scoring.dense import DenseScorer


def headline_ids(path: Path) -> set[str]:
    """The pre-registered slice, read off the flag rather than a second file."""
    with open(path, encoding="utf-8") as f:
        return {d["context_id"] for d in map(json.loads, f) if d["headline"]}


def verify_export(split: str, work: Path, out: Path) -> list[str]:
    """Check the export reproduces the pickles AND its own null controls.

    The round-trip half catches a lossy writer. The null-control half is the
    stronger claim: it says the exported files ALONE are enough to land on the
    floors, which is exactly what a consumer who never sees our pickles needs
    to be true.
    """
    failures: list[str] = []
    manifest = json.loads(
        (out / f"locus_manifest_{split}.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema") != SCHEMA:
        failures.append(f"manifest schema {manifest.get('schema')} != {SCHEMA}")

    for name, meta in manifest["files"].items():
        got = hashlib.sha256((out / name).read_bytes()).hexdigest()
        if got != meta["sha256"]:
            failures.append(f"{name}: sha256 {got[:12]} != {meta['sha256'][:12]}")

    exported_rank = load_rank(out / f"locus_rank_{split}.jsonl")
    exported_swap = load_swap(out / f"locus_swap_{split}.jsonl")
    with open(work / f"rank_items_{split}.pkl", "rb") as f:
        if exported_rank != pickle.load(f):
            failures.append("rank items do not round-trip")
    with open(work / f"swap_pairs_{split}.pkl", "rb") as f:
        if exported_swap != pickle.load(f):
            failures.append("swap pairs do not round-trip")

    anchors_path = out / f"locus_anchors_{split}.jsonl"
    exported_anchors = load_anchors(anchors_path)
    index = load_index(work / f"index_{split}.pkl")
    expected = {}
    for recs in index.values():
        golds = gold_sets(recs)
        for rec in recs:
            expected[rec.context_id] = frozenset(anchors_union(rec, golds))
    if exported_anchors != expected:
        n = sum(1 for k, v in expected.items() if exported_anchors.get(k) != v)
        failures.append(f"anchor sets differ on {n} contexts")

    # The null controls, computed from the exported files and nothing else.
    counts = load_target_counts(anchors_path)
    try:
        floor = rank_floor(pool_size_of(exported_rank))
    except ValueError as exc:
        # A truncated download lands here. Returning it as a failure keeps
        # `--verify` a report rather than a traceback.
        return [*failures, str(exc)]
    if repr(floor) != manifest["floors"]["rank_constant_mrr"]:
        failures.append(
            f"manifest floor {manifest['floors']['rank_constant_mrr']} is not "
            f"the floor of the exported pools ({floor!r})"
        )
    # Lemma 1 pins every POOL to the floor, and that is what is checked here.
    # The paper-level mean of those pools is a weaker check, not a stronger
    # one: `math.fsum([h]*n)/n` is not bit-identical to `h` for every n even
    # though the exact mean is, because the final division rounds. Measured
    # over group sizes 1..2000 it differs for 53 of them at H_10/10, 83 at
    # H_9/9 and 423 at H_5/5. So a mismatch on the aggregate alone is a fact
    # about the aggregator; a mismatch on any single pool is a defect in the
    # benchmark, and only the second is a failure.
    mrr, values, _ = rank_mrr(exported_rank, constant_scorer(1.0))
    off = {v for v in values if v != floor}
    if off:
        failures.append(
            f"rank constant null is not pinned: {len(off)} distinct pool "
            f"values off the floor {floor!r} (e.g. {sorted(off)[:3]}); a "
            "context-independent scorer must tie every pool"
        )
    elif mrr != floor:
        print(f"  note: every pool is exactly {floor!r} but their paper-level "
              f"mean is {mrr!r} -- final-division rounding in the aggregator, "
              "not a benchmark defect")
    pop_mrr, _, _ = rank_mrr(exported_rank, popularity_scorer(counts))
    if pop_mrr == floor:
        failures.append("rank popularity null is stuck on the floor")
    for name, scorer in (("popularity", popularity_scorer(counts)),
                         ("constant", constant_scorer(1.0))):
        acc, outcomes, _ = swap_accuracy(exported_swap, scorer)
        # 0.5 is exactly representable, so here the aggregate carries no
        # rounding of its own and both halves can be asserted.
        if not (acc == 0.5 and set(outcomes) == {0.5}):
            failures.append(
                f"swap {name} null = {acc!r} over the export, expected "
                "exactly 0.5 on every pair"
            )
    return failures


@dataclasses.dataclass(frozen=True)
class Metric:
    """One reported row: base, rescored, their difference and its interval.

    The CLI prints these at 5 decimal places, which is the precision the paper
    quotes, so that is also the precision anything asserting on them should
    use -- see `format_row`.
    """

    base: float
    rescored: float
    delta: float
    ci_lo: float
    ci_hi: float


def format_row(name: str, m: Metric) -> str:
    return (f"{name:6s} {m.base:9.5f} {m.rescored:9.5f} {m.delta:+9.5f}   "
            f"[{m.ci_lo:+.5f}, {m.ci_hi:+.5f}]")


def primary_result(
    export: Path,
    embeddings: Path,
    graph: Path,
    split: str = "test",
    alpha: float = 0.5,
    min_count: int = 1,
    lam: float = 4.0,
) -> dict[str, Metric]:
    """Recompute the paper's primary result from the export alone.

    Separated from `main` so the equivalence gate can call it and assert on
    the numbers directly: everything the headline depends on -- the export
    loaders, `DenseScorer`, `CoGraph.ppmi`, `rescore`, the probes and the
    cluster bootstrap -- runs inside this function and nowhere else.
    """
    rank_path = export / f"locus_rank_{split}.jsonl"
    items = load_rank(rank_path)
    keep = headline_ids(rank_path)
    anchors = load_anchors(export / f"locus_anchors_{split}.jsonl")
    print(f"{len(keep)} headline pools of {len(items)}; "
          f"{len(anchors)} anchor rows")

    scorer = DenseScorer(embeddings)
    ppmi = CoGraph.load(graph).ppmi(alpha=alpha, min_count=min_count)

    rows = {"MRR": ([], []), "R@1": ([], [])}
    groups: list[str] = []
    for it in items:
        if it.context_id not in keep:
            continue
        base = scorer.pool_scores(it.context_id, it.candidates)
        resc = rescore(base, anchors[it.context_id], ppmi, lam)
        rows["MRR"][0].append(reciprocal_rank(base, it.gold))
        rows["MRR"][1].append(reciprocal_rank(resc, it.gold))
        rows["R@1"][0].append(hit_at_1(base, it.gold))
        rows["R@1"][1].append(hit_at_1(resc, it.gold))
        groups.append(it.citing_id)

    out: dict[str, Metric] = {}
    for name, (b, a) in rows.items():
        mb, ma = paper_mean(b, groups), paper_mean(a, groups)
        diffs = [x - y for x, y in zip(a, b, strict=True)]
        lo, hi = cluster_ci(diffs, groups, n_boot=1000, seed=config.SEED)
        out[name] = Metric(base=mb, rescored=ma, delta=ma - mb,
                           ci_lo=lo, ci_hi=hi)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test")
    ap.add_argument("--export", type=Path,
                    default=config.WORK_DIR / "export")
    ap.add_argument("--embeddings", type=Path,
                    default=config.WORK_DIR / "embeddings_test.parquet")
    ap.add_argument("--graph", type=Path,
                    default=config.WORK_DIR / "cocitation_train.npz")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--lam", type=float, default=4.0)
    args = ap.parse_args(argv)

    metrics = primary_result(
        export=args.export, embeddings=args.embeddings, graph=args.graph,
        split=args.split, alpha=args.alpha, min_count=args.min_count,
        lam=args.lam,
    )
    print(f"\n{'':6s} {'base':>9s} {'+rescore':>9s} {'delta':>9s}   95% CI")
    for name, m in metrics.items():
        print(format_row(name, m))
    return 0


if __name__ == "__main__":
    sys.exit(main())
