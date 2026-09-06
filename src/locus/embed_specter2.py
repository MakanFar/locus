"""SPECTER2 embedding of the LOCUS embed-input parquets. Runs on the GPU box.

**This module needs S3 and AWS credentials, and will not run without them.**
It is not a general-purpose embedder: it downloads the model from S3, stages
its input parquet from S3, uploads its output shards to S3 and writes its
report there too, via `boto3` (declared in the optional `cloud` extra, not in
the base dependencies). A local, credential-free replacement is planned; until
it lands, produce the embeddings parquet with your own tooling.

The parquet is the actual contract, and it is small. Every LOCUS entry point
that consumes embeddings -- `locus.eval.verify`, `locus.eval.sweep`,
`locus.scoring.dense` -- wants one file (or a directory of `*.parquet` shards,
read in sorted name order) with exactly three columns:

    key        str                     context id, or paper id
    kind       str                     "context" or "paper"
    embedding  Array(Float32, dim)     the vector

`embedding` must be a polars `Array` -- an Arrow fixed-size list, one width
for the whole file -- and not a variable-length `List`, which `load_embeddings`
rejects with an IndexError. `Array(Float64, dim)` is read too; it is cast to
float32 for the cosine either way. In polars that column is built as
`pl.Series(vectors, dtype=pl.Array(pl.Float32, dim))`.

`kind` says which id space `key` lives in: `"context"` rows are keyed by
`context_id` and hold the vector of that context's masked window, `"paper"`
rows are keyed by paper id and hold the vector of that paper's title+abstract.
The dimension is inferred from the data rather than fixed, which is how HAtten
runs at 200 dims against SPECTER2's 768. Vectors are L2-normalised at load, so
their scale here does not matter, but a zero vector is a hard error. The keys
you need to cover are exactly what `locus.build.embed_inputs` collects.

Numerically this is a pinned production recipe, previously verified against a
separate CPU/GPU comparison run: allenai/specter2_base @ HF revision
3447645e, plain AutoModel, 512-token truncation, CLS pooling, fp16 on CUDA.
That comparison measured fp16-on-GPU against fp32-on-CPU at ~5e-5 cosine
agreement, so the dtype is not a risk here either.

This does its own S3 I/O and treats SageMaker's output tarball as vestigial:
what has to match the earlier verification is the numerics, not the file
handling.

Both splits are embedded in one invocation -- 233,743 rows is about ten
minutes of A10G, so sharding and resume would cost more complexity than the
job is long.

--model-s3 has no default: it names a private artefact store, which differs
per deployment and must never be baked into this public repo. Point it at
wherever you have staged the pinned allenai/specter2_base snapshot.

Usage (on SageMaker, via a private launcher not included in this repo):
    python -m locus.embed_specter2 \
        --input-s3 s3://.../locus/<run_id>/inputs \
        --output-s3 s3://.../locus/<run_id>/embeddings \
        --model-s3 s3://<your-bucket>/<your-prefix>/specter2_base \
        --splits test,val
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

MAX_LENGTH = 512
EMBED_DIM = 768


def _split_s3(uri: str) -> tuple[str, str]:
    rest = uri.removeprefix("s3://")
    bucket, _, key = rest.partition("/")
    return bucket, key


def fetch_model_dir(model_s3: str, local_dir: Path, session) -> Path:
    """Download the pinned artefact unless it is already on local disk."""
    if (local_dir / "pytorch_model.bin").exists() and (local_dir / "config.json").exists():
        return local_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    bucket, prefix = _split_s3(model_s3.rstrip("/") + "/")
    s3 = session.client("s3")
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            name = obj["Key"].removeprefix(prefix)
            if name:
                s3.download_file(bucket, obj["Key"], str(local_dir / name))
    return local_dir


def load_model(model_dir: Path, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    try:
        model = AutoModel.from_pretrained(str(model_dir), dtype=dtype)
    except TypeError:  # transformers < 4.56 spells it torch_dtype
        model = AutoModel.from_pretrained(str(model_dir), torch_dtype=dtype)
    model.eval().to(device)
    return tokenizer, model


def embed_texts(texts: list[str], tokenizer, model, device: str,
                batch_size: int, dtype=np.float32) -> np.ndarray:
    """CLS-pool every text, longest batches first so an OOM happens immediately.

    Returns vectors in the order `texts` was given, not in the order they were
    embedded: the length sort is a throughput trick and must not leak into the
    output, which is keyed positionally against the input frame.
    """
    import torch

    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    out = np.empty((len(texts), EMBED_DIM), dtype=dtype)
    done = 0
    t0 = time.time()
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = [texts[i] for i in idx]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            cls = model(**inputs).last_hidden_state[:, 0, :].float().cpu().numpy()
        out[idx] = cls.astype(dtype, copy=False)
        done += len(idx)
        if start % (batch_size * 50) == 0:
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  {done}/{len(texts)} ({rate:.0f} texts/s)", flush=True)
    return out


def check(vectors: np.ndarray, split: str) -> dict:
    """Refuse to ship a matrix that would silently corrupt every score."""
    n_nan = int(np.isnan(vectors).any(axis=1).sum())
    norms = np.linalg.norm(vectors, axis=1)
    n_zero = int((norms == 0).sum())
    if n_nan or n_zero:
        raise SystemExit(
            f"{split}: {n_nan} rows contain NaN and {n_zero} are zero vectors; "
            "a zero vector has undefined cosine and a NaN one poisons its pool"
        )
    return {
        "rows": int(vectors.shape[0]),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "norm_mean": float(norms.mean()),
    }


def run_split(split: str, input_s3: str, output_s3: str, tokenizer, model,
              device: str, batch_size: int, session,
              out_dtype: str = "float32", shard_size: int = 250_000) -> dict:
    """Embed one split in shards, writing one parquet per shard.

    Sharded because the corpus-wide completion candidate space is 1.55M rows and a
    single-pass build does not fit in a g5.xlarge's 16 GB: the input texts,
    a float32 result matrix, its float16 copy and the polars column are all
    live at once, roughly 11 GB before torch's own host allocations. Peak is
    now set by shard_size rather than by the size of the job.

    The input is staged to local disk so each shard can be sliced off lazily
    instead of holding 1.55M strings in memory for the whole run.
    """
    import io
    import os

    store = np.float16 if out_dtype == "float16" else np.float32
    cell = pl.Float16 if out_dtype == "float16" else pl.Float32

    bucket, key = _split_s3(f"{input_s3.rstrip('/')}/embed_inputs_{split}.parquet")
    local = f"/tmp/embed_inputs_{split}.parquet"
    session.client("s3").download_file(bucket, key, local)
    n_rows = int(pl.scan_parquet(local).select(pl.len()).collect().item())
    n_shards = -(-n_rows // shard_size)
    print(f"[{split}] {n_rows} rows in {n_shards} shard(s) of {shard_size}",
          flush=True)

    stats = {"rows": 0, "norm_min": float("inf"), "norm_max": 0.0,
             "norm_sum": 0.0, "shards": []}
    for shard in range(n_shards):
        part = (pl.scan_parquet(local)
                .slice(shard * shard_size, shard_size)
                .collect())
        vectors = embed_texts(part["text"].to_list(), tokenizer, model, device,
                              batch_size, dtype=store)
        norms = np.linalg.norm(vectors.astype(np.float32), axis=1)
        n_nan = int(np.isnan(vectors).any(axis=1).sum())
        n_zero = int((norms == 0).sum())
        if n_nan or n_zero:
            raise SystemExit(
                f"{split} shard {shard}: {n_nan} rows contain NaN and {n_zero} "
                "are zero vectors; a zero vector has undefined cosine and a "
                "NaN one poisons its pool"
            )
        stats["rows"] += int(vectors.shape[0])
        stats["norm_min"] = min(stats["norm_min"], float(norms.min()))
        stats["norm_max"] = max(stats["norm_max"], float(norms.max()))
        stats["norm_sum"] += float(norms.sum())

        # A flat Series reshaped into the Array dtype, never list(vectors):
        # that builds one numpy object per row, 1.55M of them.
        col = pl.Series("embedding", vectors.reshape(-1), dtype=cell).reshape(
            (vectors.shape[0], EMBED_DIM))
        out = part.select("key", "kind").with_columns(embedding=col)
        buf = io.BytesIO()
        out.write_parquet(buf)
        size = buf.tell()
        buf.seek(0)
        name = f"embeddings_{split}_shard{shard:03d}.parquet"
        ob, ok = _split_s3(f"{output_s3.rstrip('/')}/{name}")
        session.client("s3").upload_fileobj(buf, ob, ok)
        stats["shards"].append(name)
        print(f"[{split}] shard {shard + 1}/{n_shards} -> s3://{ob}/{ok} "
              f"({size / 1e6:.0f} MB)", flush=True)
        del part, vectors, col, out, buf

    os.remove(local)
    stats["norm_mean"] = stats["norm_sum"] / max(stats["rows"], 1)
    del stats["norm_sum"]
    return {"split": split, **stats}


def main(argv: list[str] | None = None) -> int:
    import boto3
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-s3", required=True)
    ap.add_argument("--output-s3", required=True)
    ap.add_argument("--splits", default="test,val")
    ap.add_argument(
        "--model-s3", required=True,
        help="s3://<your-bucket>/<your-prefix> holding the pinned specter2_base snapshot "
             "(private; no public default -- see the module docstring)",
    )
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--model-dir", default="/tmp/specter2_base")
    ap.add_argument("--out-dtype", choices=("float32", "float16"), default="float32")
    ap.add_argument("--shard-size", type=int, default=250_000)
    args = ap.parse_args(argv)

    session = boto3.Session()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__}", flush=True)

    model_dir = fetch_model_dir(args.model_s3, Path(args.model_dir), session)
    tokenizer, model = load_model(model_dir, device)

    report = {
        "model_s3": args.model_s3,
        "device": device,
        "batch_size": args.batch_size,
        "max_length": MAX_LENGTH,
        "pooling": "CLS",
        "dtype": "float16" if device == "cuda" else "float32",
        "out_dtype": args.out_dtype,
        "splits": [],
    }
    for split in args.splits.split(","):
        report["splits"].append(
            run_split(split.strip(), args.input_s3, args.output_s3, tokenizer,
                      model, device, args.batch_size, session, args.out_dtype,
                      args.shard_size)
        )

    ob, ok = _split_s3(f"{args.output_s3.rstrip('/')}/embed_report.json")
    session.client("s3").put_object(
        Bucket=ob, Key=ok, Body=json.dumps(report, indent=2).encode()
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
