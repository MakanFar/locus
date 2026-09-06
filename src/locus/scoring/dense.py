"""A LOCUS base scorer over precomputed bi-encoder embeddings.

S_base(c | l) = cosine(v(l), v(c)) -- the location's masked window against the
candidate paper's title+abstract vector. This is the P(c | l) side of the v2
spine; the structural P(c | l, A) side is rescore.py.

Vectors are L2-normalised once at load, so a score is a dot product and the
cosine denominator can never be recomputed inconsistently between call sites.
`pool_scores` exists because the lambda sweep scores the same pool at many
lambdas: doing it per candidate through __call__ would repeat 554,880 dot
products for every lambda, where the base pool only has to be built once.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

CHUNK = 65_536


def _check_embedding_dtype(shard: Path, schema) -> None:
    """Refuse an `embedding` column that is not a fixed-width `Array`.

    This is the single most likely mistake a bring-your-own-model user makes:
    building the column from a Python list of vectors lets polars infer
    `List(Float32)` -- a variable-length column -- which reads back from
    parquet as a 1-D object array of arrays. `block.shape[1]` on that raised
    `IndexError: tuple index out of range`, naming neither the column nor the
    cause, several frames below the code the user wrote. README.md claimed
    the loader rejected it; this is the code that makes that true.

    A `List` column can also be ragged, which no reshape could rescue: one
    row of 768 and one of 767 is not a matrix, and silently accepting the
    well-formed case would leave the ragged one to fail somewhere further in.
    """
    dtype = schema.get("embedding")
    if dtype is None:
        raise SystemExit(
            f"{shard}: no 'embedding' column; an embeddings parquet needs "
            "key (str), kind (str) and embedding (Array(Float32, dim))"
        )
    if not isinstance(dtype, pl.Array):
        raise SystemExit(
            f"{shard}: column 'embedding' has dtype {dtype}, which is not a "
            "fixed-width Arrow array. It must be Array(Float32, dim) -- one "
            "width for the whole file. In polars, build it with "
            "pl.Series(matrix, dtype=pl.Array(pl.Float32, dim)); passing a "
            "list of vectors instead infers List(...), which is "
            "variable-length and has no single dimension to score against."
        )


def load_embeddings(path: str | Path) -> dict[str, tuple[dict[str, int], np.ndarray]]:
    """Return {kind: (key -> row, L2-normalised matrix)} from an embeddings parquet.

    Normalisation is chunked and preserves the stored dtype. The completion experiment's
    candidate space is 1.48M papers, where the previous `.to_list()` would
    have built 1.48M Python lists before numpy ever saw them -- tens of GB on
    a 16 GB machine. `.to_numpy()` on an Array column is also 100x faster on
    the small artefacts, and returns the identical matrix.
    """
    # A directory means a sharded build (see embed_specter2.run_split); shards
    # are read in sorted name order so row order is reproducible.
    #
    # Shards are read ONE AT A TIME into a preallocated matrix, never
    # concatenated. pl.concat over the 1.55M-row corpus-wide build holds
    # every shard plus the joined frame at once -- about 4.5 GB before the
    # numpy copy -- which drove a 16 GB machine into swap. Peak here is the
    # final matrix plus one shard.
    path = Path(path)
    shards = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if path.is_dir() and not shards:
        raise SystemExit(f"{path} is a directory but contains no .parquet shards")

    keys: dict[str, list[str]] = {"context": [], "paper": []}
    counts: dict[str, int] = {"context": 0, "paper": 0}
    dtype = None
    for shard in shards:
        meta = pl.read_parquet(shard, columns=["key", "kind"])
        for kind in ("context", "paper"):
            sel = meta.filter(pl.col("kind") == kind)["key"].to_list()
            keys[kind].extend(sel)
            counts[kind] += len(sel)
    for kind in ("context", "paper"):
        if counts[kind] == 0:
            raise SystemExit(f"{path} contains no rows of kind {kind!r}")

    mats: dict[str, np.ndarray] = {}
    filled = {"context": 0, "paper": 0}
    for shard in shards:
        frame = pl.read_parquet(shard)
        _check_embedding_dtype(shard, frame.schema)
        block_all = frame["embedding"].to_numpy()
        kinds = frame["kind"].to_numpy()
        for kind in ("context", "paper"):
            take = kinds == kind
            n = int(take.sum())
            if not n:
                continue
            block = block_all[take]
            if kind not in mats:
                dtype = block.dtype
                mats[kind] = np.empty((counts[kind], block.shape[1]), dtype=dtype)
            mats[kind][filled[kind] : filled[kind] + n] = block
            filled[kind] += n
        del frame, block_all

    out: dict[str, tuple[dict[str, int], np.ndarray]] = {}
    for kind in ("context", "paper"):
        mat = mats[kind]
        for i in range(0, len(mat), CHUNK):
            block = mat[i : i + CHUNK].astype(np.float32)
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            if not np.all(norms > 0):
                raise SystemExit(
                    f"{path}: {int((norms == 0).sum())} {kind} vectors have zero "
                    "norm, whose cosine is undefined"
                )
            mat[i : i + CHUNK] = (block / norms).astype(mat.dtype)
        out[kind] = ({k: i for i, k in enumerate(keys[kind])}, mat)
    return out


class DenseScorer:
    """Callable (context_id, candidate_id) -> cosine, plus a batched pool form."""

    def __init__(self, path: str | Path):
        parts = load_embeddings(path)
        self._ctx_ix, self._ctx = parts["context"]
        self._doc_ix, self._doc = parts["paper"]

    def __call__(self, context_id: str, candidate_id: str) -> float:
        try:
            q = self._ctx[self._ctx_ix[context_id]]
            d = self._doc[self._doc_ix[candidate_id]]
        except KeyError as exc:
            # Silently returning 0.0 would tie that candidate against the pool
            # mean and look like a merely weak score rather than a missing one.
            raise SystemExit(f"no embedding for {exc.args[0]!r}") from exc
        return float(np.dot(q.astype(np.float32), d.astype(np.float32)))

    def pool_scores(self, context_id: str, candidates) -> dict[str, float]:
        try:
            q = self._ctx[self._ctx_ix[context_id]]
            rows = [self._doc_ix[c] for c in candidates]
        except KeyError as exc:
            raise SystemExit(f"no embedding for {exc.args[0]!r}") from exc
        sims = self._doc[rows].astype(np.float32) @ q.astype(np.float32)
        return {c: float(s) for c, s in zip(candidates, sims, strict=True)}
