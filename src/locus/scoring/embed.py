"""A local, credential-free SPECTER2 embedder for anyone with a GPU or a laptop.

`locus.embed_specter2` is the cloud path: it needs S3, AWS credentials and a
staged model snapshot, and stays that way. This module exists so a reviewer
who cannot reach that infrastructure can still produce vectors that every
LOCUS entry point consuming embeddings -- `locus.eval.verify`,
`locus.eval.sweep`, `locus.scoring.dense` -- accepts: `key`, `kind`,
`embedding`, where `embedding` is a polars `Array(Float32, dim)` (an Arrow
fixed-size list), never a variable-length `List`. A `List` column loads fine
in polars and fails deep inside `load_embeddings` with an `IndexError` that
names neither the column nor the reason -- so the shape is enforced here at
the point of construction with `pl.Series(vectors, dtype=pl.Array(pl.Float32,
dim))`, not left to whatever polars infers from a list of numpy arrays.

Numerically this reproduces the pinned production recipe documented in
`locus.embed_specter2`: allenai/specter2_base at HF revision 3447645e, plain
`AutoModel`, 512-token truncation, CLS pooling
(`last_hidden_state[:, 0, :]`), fp16 on CUDA and fp32 everywhere else. None of
that is a choice made here -- it is copied, because the published vectors
were produced with exactly this recipe and changing any part of it would
mean this module produces different embeddings than the ones already on
disk.

`torch` and `transformers` are imported inside functions, never at module
scope: they live in the `gpu` extra, not in the base install, and the test
suite (and CI, which installs only `.[dev,node2vec]`) must collect and pass
without them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

MODEL_REVISION = "3447645e"
MAX_LENGTH = 512
EMBED_DIM = 768

MODEL_REVISIONS: dict[str, str] = {
    # The revision the published SPECTER2 vectors were produced at.
    "allenai/specter2_base": MODEL_REVISION,
    # SciNCL, the second citation-trained base in the paper. Pinned to the
    # repository's 2024-06-04 commit, the latest at the time of writing; the
    # SciNCL vectors in the paper were produced from that snapshot.
    "malteos/scincl": "ebc5348d184ba2fc9beee69b4e394263fce57b2e",
}
"""Per-model Hugging Face revision pins.

One pin cannot serve every `--model`: SPECTER2's commit hash does not exist in
SciNCL's repository, and passing it there fails inside transformers with a
message about the wrong repository rather than about the pin. A model with no
entry here is loaded at the hub's current head, with a warning, because that
is a reproducibility gap the user should see rather than a reason to refuse
an encoder the benchmark can perfectly well score.
"""

_HEX = frozenset("0123456789abcdef")


def revision_for(model: str, override: str | None = None) -> str | None:
    """The revision to load `model` at: an explicit override, else the pin.

    Returns None for an unpinned model with no override, after saying so on
    stderr. An override must look like a commit-hash prefix -- a branch name
    such as "main" moves under you and would defeat the point of pinning.
    """
    if override is not None:
        if len(override) < 7 or not set(override) <= _HEX:
            raise ValueError(
                f"revision {override!r} is not a commit-hash prefix (>= 7 hex "
                "characters); branch or tag names are not accepted because "
                "they move"
            )
        return override
    pinned = MODEL_REVISIONS.get(model)
    if pinned is None:
        print(
            f"warning: {model!r} is not pinned in locus.scoring.embed."
            "MODEL_REVISIONS; loading the hub's current revision, which is "
            "not reproducible -- pass --revision <commit> to fix it",
            file=sys.stderr,
        )
    return pinned


def pick_device() -> str:
    """cuda -> mps -> cpu, in order of how much of the recipe they preserve.

    fp16 is only used on cuda (see `embed_frame`); mps and cpu both run fp32,
    which is why the frozen fp16-on-GPU vectors are compared against
    fp32-on-CPU in the reproduction check rather than against anything
    fp16.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_frame(
    frame: pl.DataFrame,
    model: str,
    *,
    device: str | None = None,
    batch_size: int = 32,
    max_length: int = MAX_LENGTH,
    revision: str | None = None,
) -> pl.DataFrame:
    """Embed `frame["text"]`, returning `key`, `kind`, `embedding`.

    `embedding` is a fixed-size `Array(Float32, dim)`, built directly from
    the result matrix with `pl.Series(vectors, dtype=pl.Array(pl.Float32,
    dim))` -- not `list(vectors)`, which would let polars infer a
    variable-length `List` instead.

    `revision` overrides the per-model pin in `MODEL_REVISIONS`; see
    `revision_for`.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = pick_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    rev = revision_for(model, revision)

    tokenizer = AutoTokenizer.from_pretrained(model, revision=rev)
    try:
        net = AutoModel.from_pretrained(model, revision=rev, dtype=dtype)
    except TypeError:  # transformers < 4.56 spells it torch_dtype -- the
        # sibling module locus.embed_specter2 carries the same fallback for
        # the same reason: pyproject.toml's gpu extra declares
        # transformers>=4.40, and 4.40-4.55 raise a bare TypeError on the
        # `dtype=` kwarg. Do not delete this as redundant just because the
        # machine you're testing on happens to have a newer transformers.
        net = AutoModel.from_pretrained(model, revision=rev, torch_dtype=dtype)
    net.eval().to(device)

    texts = frame["text"].to_list()
    dim = net.config.hidden_size
    out = np.empty((len(texts), dim), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch, padding=True, truncation=True, max_length=max_length,
                return_tensors="pt",
            ).to(device)
            cls = net(**inputs).last_hidden_state[:, 0, :]
            out[start : start + len(batch)] = cls.float().cpu().numpy()

    return frame.select("key", "kind").with_columns(
        embedding=pl.Series(out, dtype=pl.Array(pl.Float32, dim))
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True,
                     help="an embed-inputs parquet: key, kind, text")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="allenai/specter2_base",
                    help="any Hugging Face bi-encoder; pinned revisions exist "
                         f"for {sorted(MODEL_REVISIONS)}")
    ap.add_argument("--revision", default=None,
                    help="commit hash to load --model at; overrides the pin "
                         "and is required for reproducibility with an "
                         "unpinned model")
    ap.add_argument("--device", default=None, choices=(None, "cuda", "mps", "cpu"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = ap.parse_args(argv)

    frame = pl.read_parquet(args.input)
    out = embed_frame(
        frame, args.model, device=args.device, batch_size=args.batch_size,
        max_length=args.max_length, revision=args.revision,
    )

    norms = np.linalg.norm(out["embedding"].to_numpy(), axis=1)
    n_nan = int(np.isnan(out["embedding"].to_numpy()).any(axis=1).sum())
    n_zero = int((norms == 0).sum())
    if n_nan or n_zero:
        raise SystemExit(
            f"{n_nan} rows contain NaN and {n_zero} are zero vectors; a zero "
            "vector has undefined cosine and a NaN one poisons its pool"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.output)
    print(f"{out.height} vectors written to {args.output} "
          f"(dim {out.schema['embedding'].size}, norms "
          f"{norms.min():.3f}-{norms.max():.3f}, mean {norms.mean():.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
