"""Fixtures shared across the test suite."""
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from locus.core.types import ContextRec
from locus.scoring.cocitation import count_sites
from locus.scoring.dense import DenseScorer
from locus.scoring.node2vec import NodeVectors


def _rec(cid, refid, loc):
    return ContextRec(
        context_id=f"{cid}_{refid}_{loc}",
        citing_id=cid,
        refid=refid,
        raw="",
        masked="",
        location=loc,
    )


@pytest.fixture
def second_corpus() -> Path:
    """The synthetic second corpus (see
    tests/data/fixtures/second_corpus/generate.py).

    Not derived from Gu et al.'s data -- committed JSONL, built and
    self-checked by the generator script committed beside it. Lives here
    rather than in tests/data/conftest.py because the custom-corpus path it
    exercises runs through `locus.build` as well as `locus.data`.
    """
    return Path(__file__).parent / "data" / "fixtures" / "second_corpus"


@pytest.fixture
def tiny_graph():
    """A minimal train co-citation graph: a/b co-cited twice, a/c once."""
    sites = [
        ("p1", [_rec("p1", "a", 0), _rec("p1", "b", 0)]),
        ("p2", [_rec("p2", "a", 0), _rec("p2", "b", 0)]),
        ("p3", [_rec("p3", "a", 0), _rec("p3", "c", 0)]),
    ]
    return count_sites(sites)


@pytest.fixture
def tiny_vectors():
    """A minimal NodeVectors view over 3 papers."""
    v = np.array([[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0]], dtype=np.float32)
    return NodeVectors(vectors=v, index={"c": 0, "a1": 1, "a2": 2})


@pytest.fixture
def dense_scorer(tmp_path):
    """A tiny on-disk DenseScorer, built the same way tests/test_dense.py's
    _write/_basic helpers do: one context and one paper vector, parallel to
    each other so cosine similarity is exactly 1.0."""
    dim = 4
    rows = [
        ("ctx1", "context", [1.0, 0.0, 0.0, 0.0]),
        ("p_x", "paper", [2.0, 0.0, 0.0, 0.0]),  # parallel to ctx1
    ]
    frame = pl.DataFrame(
        {"key": [r[0] for r in rows], "kind": [r[1] for r in rows]}
    ).with_columns(
        embedding=pl.Series(
            [np.asarray(r[2], dtype=np.float32) for r in rows],
            dtype=pl.Array(pl.Float32, dim),
        )
    )
    path = tmp_path / "e.parquet"
    frame.write_parquet(path)
    return DenseScorer(path)


@pytest.fixture(scope="session")
def exported_bundle():
    """The local export directory, for tests that read released files."""
    import os

    work = os.environ.get("LOCUS_WORK_DIR")
    if not work:
        pytest.skip("needs LOCUS_WORK_DIR")
    path = Path(work) / "export"
    if not path.exists():
        pytest.skip(f"no export at {path}")
    return path
