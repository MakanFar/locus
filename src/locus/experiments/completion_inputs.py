"""Build the sequential-completion candidate space and its embedding inputs.

Sequential completion completes a citation set: at a location with |G(l)| >= 2, reveal
its citations one at a time and try to recover the rest. The candidate space
is the shipped oracle prefetch, not corpus-wide retrieval --
otherwise compounding error is uninterpretable.

Measured on test: 22,274 locations qualify, spread over 69,874 contexts and
6,766 citing papers, and their prefetch lists name **1,481,171 distinct
papers** -- 89% of the corpus. The prefetch is genuinely oracle: all 69,874
contexts contain their own gold, 0 violations.

The candidate space for a location is the UNION of its contexts' lists rather
than any single list. A single list covers only 83.02% of G(l), which would
cap Recall@K below 1 for reasons that have nothing to do with the method; the
union covers 100.00%, so the ceiling is the metric's own.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from locus import config
from locus.build.embed_inputs import compose_paper_text
from locus.core.types import gold_sets
from locus.data.corpus import load_index, load_papers


def prefetch_path() -> Path:
    """Resolved on call, not at import: LOCUS_DATA_DIR is required, and a
    module that raises on import cannot even be listed by `--help`."""
    return config.data_dir() / "test_with_oracle_prefetched_ids_for_reranking.json"


def locations_with_sets(split: str, min_gold: int = 2):
    """(citing_id, location) -> gold set, and context_id -> that key."""
    index = load_index(config.WORK_DIR / f"index_{split}.pkl")
    loc_gold, ctx_loc, ctx_rec = {}, {}, {}
    for citing, recs in index.items():
        golds = gold_sets(recs)
        for loc, g in golds.items():
            if len(g) >= min_gold:
                loc_gold[(citing, loc)] = set(g)
        for rec in recs:
            ctx_loc[rec.context_id] = (citing, rec.location)
            ctx_rec[rec.context_id] = rec
    return loc_gold, ctx_loc, ctx_rec


def read_prefetch(wanted: set[str]) -> dict[str, list[str]]:
    out = {}
    with open(prefetch_path(), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["context_id"] in wanted:
                out[rec["context_id"]] = [str(x) for x in rec["prefetched_ids"]]
    missing = wanted - set(out)
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(wanted)} contexts have no prefetch list, "
            f"e.g. {sorted(missing)[:5]}"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test")
    ap.add_argument("--min-gold", type=int, default=2)
    args = ap.parse_args(argv)

    loc_gold, ctx_loc, ctx_rec = locations_with_sets(args.split, args.min_gold)
    wanted = {c for c, k in ctx_loc.items() if k in loc_gold}
    print(f"{len(loc_gold)} locations, {len(wanted)} contexts, "
          f"{len({k[0] for k in loc_gold})} citing papers")

    prefetch = read_prefetch(wanted)
    by_loc: dict[tuple[str, int], set[str]] = {}
    for cid, ids in prefetch.items():
        by_loc.setdefault(ctx_loc[cid], set()).update(ids)
    every_id = set().union(*by_loc.values())
    print(f"{len(every_id)} distinct candidate papers")

    reach = sum(len(g & by_loc[k]) for k, g in loc_gold.items() if k in by_loc)
    total = sum(len(g) for k, g in loc_gold.items() if k in by_loc)
    print(f"G(l) reachable in the union: {reach}/{total} ({100 * reach / total:.2f}%)")

    # JSONL, one location per line, NOT a pickled list. The candidate lists
    # total 47,895,173 id strings across the 22,274 locations; unpickled in
    # one go that is about 3.1 GB of Python str objects, which is what drove
    # this machine into swap. Streamed, only one location is resident.
    task_path = config.WORK_DIR / f"expb_task_{args.split}.jsonl"
    n = 0
    with open(task_path, "w", encoding="utf-8") as f:
        for k, g in sorted(loc_gold.items()):
            if k not in by_loc:
                continue
            f.write(json.dumps({
                "citing_id": k[0], "location": k[1], "gold": sorted(g),
                "candidates": sorted(by_loc[k]),
                "contexts": sorted(c for c, kk in ctx_loc.items() if kk == k),
            }) + "\n")
            n += 1
    print(f"wrote {task_path} ({n} locations)")

    papers = load_papers(every_id)
    rows = [{"key": pid, "kind": "paper",
             "text": compose_paper_text(papers[pid]["title"], papers[pid]["abstract"])}
            for pid in sorted(every_id)]
    rows += [{"key": cid, "kind": "context", "text": ctx_rec[cid].masked}
             for cid in sorted(wanted)]
    frame = pl.DataFrame(rows, schema={"key": pl.Utf8, "kind": pl.Utf8,
                                       "text": pl.Utf8})
    out = config.WORK_DIR / f"embed_inputs_expb_{args.split}.parquet"
    frame.write_parquet(out)
    print(f"{frame.height} rows -> {out} ({out.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
