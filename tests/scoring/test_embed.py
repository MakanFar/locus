"""The local embedder's output must be loadable by the scorer that consumes it.

The failure this guards against is silent at write time and cryptic at read
time: a variable-length polars List column raises IndexError deep inside
load_embeddings, naming neither the column nor the reason.
"""
import polars as pl
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from locus.scoring.dense import load_embeddings
from locus.scoring.embed import embed_frame


def test_output_is_a_fixed_size_array_that_load_embeddings_accepts(tmp_path):
    frame = pl.DataFrame(
        {"key": ["p1", "c1"], "kind": ["paper", "context"],
         "text": ["a title [SEP] an abstract", "a masked window TARGETCIT"]},
        schema={"key": pl.Utf8, "kind": pl.Utf8, "text": pl.Utf8},
    )
    out = embed_frame(frame, "allenai/specter2_base", device="cpu", batch_size=2)
    assert isinstance(out.schema["embedding"], pl.Array)
    path = tmp_path / "e.parquet"
    out.write_parquet(path)
    loaded = load_embeddings(path)
    assert set(loaded) == {"paper", "context"}
