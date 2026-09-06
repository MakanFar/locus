"""Corpus-wide first-stage retrieval, replacing the oracle prefetch.

Sequential completion as first run reranks the dataset's *oracle* prefetch: ~2,150
candidates per location, constructed so that the gold is always present. That
is the standard reranking protocol and it measures reranking, but it is not
end-to-end local citation recommendation -- there the base retriever
must go and find the candidates itself, and it is allowed to fail.

This module builds that honest candidate set: cosine over every paper in the
embedding store, top-K per location, written to a task file the completion experiment can
consume unchanged. The number that matters is first-stage recall -- what
fraction of gold survives retrieval -- because it caps everything downstream
and the prefetch protocol hides it at 100%.

**One caveat, stated because it is not removable.** The store holds 1,481,171
papers, the union of every prefetch list on the split, against a corpus of
1,661,201. The ~11% never embedded are papers no prefetch ever proposed, so
they can only ever have been distractors: retrieving over the union is
therefore mildly optimistic, never pessimistic. Reporting a corpus-wide number
that silently drops 11% of the corpus would be worse than reporting the gap.

Compute is arranged around the one hard constraint: a (queries x papers) score
matrix does not fit. Queries are chunked, and for each chunk the paper matrix
is streamed in blocks and cast to float32 a block at a time -- numpy has no
float16 BLAS, so the cast has to happen somewhere, and doing it per block
keeps the working set at one block instead of the whole 2.3 GB store.

Usage:
    .venv/bin/python -m locus.scoring.retrieve \\
        --store work/embeddings_expb_test_mm --k 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from locus import config
from locus.scoring.mmapstore import MmapScorer

QUERY_CHUNK = 4096
PAPER_BLOCK = 200_000


def merge_topk(best_s, best_i, s, i, k):
    """Append a block's candidates and trim back to the k best.

    Extracted and tested on its own because dropping it is FUNCTIONALLY
    invisible -- the final sort still returns the right k -- while turning
    the working set from O(k) into O(corpus). That is the exact shape of the
    completion-run OOM: the output was sized, the thing producing it was not.
    """
    best_s = np.concatenate([best_s, s], axis=1)
    best_i = np.concatenate([best_i, i], axis=1)
    if best_s.shape[1] > k:
        keep = np.argpartition(-best_s, k - 1, axis=1)[:, :k]
        best_s = np.take_along_axis(best_s, keep, axis=1)
        best_i = np.take_along_axis(best_i, keep, axis=1)
    return best_s, best_i


def top_k(store: MmapScorer, context_ids: list[str], k: int,
          query_chunk: int = QUERY_CHUNK, paper_block: int = PAPER_BLOCK,
          progress=None) -> np.ndarray:
    """Row i holds the store-row indices of the k nearest papers to query i.

    Vectors are unit-normalised at build time, so the dot product IS the
    cosine; no renormalisation here, and a change to that invariant in
    mmapstore.build would silently turn this into an inner-product search.
    """
    docs = store._doc
    n_docs = docs.shape[0]
    k = min(k, n_docs)
    out = np.empty((len(context_ids), k), dtype=np.int64)

    for q0 in range(0, len(context_ids), query_chunk):
        ids = context_ids[q0:q0 + query_chunk]
        rows = np.fromiter((store._ctx_ix[c] for c in ids),
                           dtype=np.int64, count=len(ids))
        q = store._ctx[rows].astype(np.float32)
        best_s = np.full((len(ids), 0), -np.inf, dtype=np.float32)
        best_i = np.zeros((len(ids), 0), dtype=np.int64)

        for p0 in range(0, n_docs, paper_block):
            block = docs[p0:p0 + paper_block].astype(np.float32)
            sims = q @ block.T
            take = min(k, sims.shape[1])
            part = np.argpartition(-sims, take - 1, axis=1)[:, :take]
            s = np.take_along_axis(sims, part, axis=1)
            best_s, best_i = merge_topk(best_s, best_i, s, part + p0, k)
            del sims, block

        order = np.argsort(-best_s, axis=1)[:, :k]
        out[q0:q0 + len(ids)] = np.take_along_axis(best_i, order, axis=1)
        if progress:
            progress(min(q0 + len(ids), len(context_ids)), len(context_ids))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test")
    ap.add_argument("--store", type=Path,
                    default=config.WORK_DIR / "embeddings_expb_test_mm")
    ap.add_argument("--k", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    src = config.WORK_DIR / f"expb_task_{args.split}.jsonl"
    out = args.out or config.WORK_DIR / f"expb_task_{args.split}_corpuswide.jsonl"
    locs = []
    with open(src, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            locs.append(json.loads(line))
    print(f"{len(locs)} locations from {src}")

    store = MmapScorer(args.store)
    papers = store._doc_ix
    inv = list(papers)
    print(f"store: {len(papers)} papers, {len(store._ctx_ix)} contexts")

    queries = [loc["contexts"][0] for loc in locs]
    missing = [c for c in queries if c not in store._ctx_ix]
    if missing:
        raise SystemExit(
            f"{len(missing)} query contexts are absent from the store "
            f"(first: {missing[0]!r}); retrieval cannot be run against a "
            f"store that does not hold the queries"
        )

    def tick(done: int, total: int) -> None:
        print(f"  retrieved {done}/{total}", flush=True)

    ranked = top_k(store, queries, args.k, progress=tick)

    # First-stage recall is the headline diagnostic: everything downstream is
    # capped by it, and the prefetch protocol pins it at 1.0 by construction.
    hits = misses = 0
    at = {1: 0, 10: 0, 100: 0, 1000: 0, args.k: 0}
    gold_total = 0
    with open(out, "w", encoding="utf-8") as f:
        for loc, row in zip(locs, ranked, strict=True):
            cands = [inv[j] for j in row]
            pos = {c: i for i, c in enumerate(cands)}
            gold_total += len(loc["gold"])
            for g in loc["gold"]:
                i = pos.get(g)
                if i is None:
                    misses += 1
                    continue
                hits += 1
                for cut in at:
                    if i < cut:
                        at[cut] += 1
            f.write(json.dumps({**loc, "candidates": cands,
                                "prefetch": "corpuswide"}) + "\n")

    print(f"\nfirst-stage recall over {gold_total} gold citations:")
    for cut in sorted(at):
        print(f"  R@{cut:<6} {at[cut] / gold_total:.5f}")
    print(f"  retrieved {hits}, missed {misses}")
    report = {
        "split": args.split, "locations": len(locs), "k": args.k,
        "store_papers": len(papers), "gold_total": gold_total,
        "first_stage_recall": {str(c): at[c] / gold_total for c in sorted(at)},
        "missed": misses,
    }
    rp = config.WORK_DIR / f"retrieve_report_{args.split}.json"
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}\nwrote {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
