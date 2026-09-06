"""A memory-mapped embedding store, for candidate spaces that exceed RAM.

Sequential completion scores 22,274 locations against a union of 1,481,171 candidate
papers. Held in RAM that matrix is 2.27 GB at float16 before the key index,
and on a 16 GB laptop already running a browser and an editor it does not
survive: an in-RAM load drove swap to 7.2 GB of 8 GB and the process stopped
making progress entirely (800 KB resident, 0% CPU).

The access pattern is what makes mmap right rather than merely smaller. Each
location touches only its own ~2,000-4,000 candidate rows, so the resident
set is the pages actually read, and the OS evicts the rest under pressure
instead of the process dying.

Vectors are stored L2-normalised at build time, so a score is a dot product
and nothing has to normalise at query time -- which would mean touching every
page on every call and defeat the point.

Build once:
    .venv/bin/python -m locus.scoring.mmapstore --shards work/embeddings_expb_test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

CHUNK = 65_536


def build(shard_dir: Path, out_prefix: Path) -> dict:
    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"{shard_dir} contains no .parquet shards")

    counts = {"context": 0, "paper": 0}
    keys: dict[str, list[str]] = {"context": [], "paper": []}
    dim = None
    for shard in shards:
        meta = pl.read_parquet(shard, columns=["key", "kind"])
        for kind in ("context", "paper"):
            sel = meta.filter(pl.col("kind") == kind)["key"].to_list()
            keys[kind].extend(sel)
            counts[kind] += len(sel)
    print(f"{counts['paper']} papers, {counts['context']} contexts", flush=True)

    handles, filled = {}, {"context": 0, "paper": 0}
    for shard in shards:
        frame = pl.read_parquet(shard)
        block_all = frame["embedding"].to_numpy()
        kinds = frame["kind"].to_numpy()
        dim = block_all.shape[1]
        for kind in ("context", "paper"):
            take = kinds == kind
            n = int(take.sum())
            if not n:
                continue
            if kind not in handles:
                handles[kind] = np.lib.format.open_memmap(
                    f"{out_prefix}_{kind}.npy", mode="w+",
                    dtype=np.float16, shape=(counts[kind], dim),
                )
            block = block_all[take].astype(np.float32)
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            if not np.all(norms > 0):
                raise SystemExit(
                    f"{shard}: {int((norms == 0).sum())} {kind} vectors have "
                    "zero norm, whose cosine is undefined"
                )
            handles[kind][filled[kind] : filled[kind] + n] = (
                block / norms).astype(np.float16)
            filled[kind] += n
        del frame, block_all
        print(f"  {shard.name}: papers {filled['paper']}, "
              f"contexts {filled['context']}", flush=True)

    for h in handles.values():
        h.flush()
    manifest = {"dim": int(dim), "counts": counts,
                "keys": {k: keys[k] for k in ("context", "paper")}}
    with open(f"{out_prefix}_index.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    return manifest


class MmapScorer:
    """DenseScorer's interface, backed by memory-mapped normalised vectors."""

    def __init__(self, prefix: str | Path):
        prefix = str(prefix)
        with open(f"{prefix}_index.json", encoding="utf-8") as f:
            manifest = json.load(f)
        self._ctx_ix = {k: i for i, k in enumerate(manifest["keys"]["context"])}
        self._doc_ix = {k: i for i, k in enumerate(manifest["keys"]["paper"])}
        self._ctx = np.load(f"{prefix}_context.npy", mmap_mode="r")
        self._doc = np.load(f"{prefix}_paper.npy", mmap_mode="r")

    def __call__(self, context_id: str, candidate_id: str) -> float:
        try:
            q = self._ctx[self._ctx_ix[context_id]]
            d = self._doc[self._doc_ix[candidate_id]]
        except KeyError as exc:
            raise SystemExit(f"no embedding for {exc.args[0]!r}") from exc
        return float(np.dot(q.astype(np.float32), d.astype(np.float32)))

    def pool_scores(self, context_id: str, candidates) -> dict[str, float]:
        try:
            q = self._ctx[self._ctx_ix[context_id]].astype(np.float32)
            rows = np.fromiter((self._doc_ix[c] for c in candidates),
                               dtype=np.int64, count=len(candidates))
        except KeyError as exc:
            raise SystemExit(f"no embedding for {exc.args[0]!r}") from exc
        # Sorted gather: mmap pages come back in file order, so a sorted read
        # touches each page once instead of walking back and forth over 2.3 GB.
        srt = np.argsort(rows, kind="stable")
        sims = np.empty(len(rows), dtype=np.float32)
        sims[srt] = self._doc[rows[srt]].astype(np.float32) @ q
        return {c: float(s) for c, s in zip(candidates, sims, strict=True)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, default=None)
    args = ap.parse_args(argv)
    prefix = args.out_prefix or args.shards.with_name(args.shards.name + "_mm")
    manifest = build(args.shards, prefix)
    print(f"wrote {prefix}_paper.npy / _context.npy / _index.json "
          f"(dim {manifest['dim']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
