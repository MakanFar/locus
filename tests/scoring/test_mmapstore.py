import numpy as np
import polars as pl
import pytest

from locus.scoring.mmapstore import MmapScorer, build

DIM = 4


def _shards(tmp_path, spec):
    d = tmp_path / "shards"
    d.mkdir()
    for si, rows in enumerate(spec):
        pl.DataFrame(
            {"key": [r[0] for r in rows], "kind": [r[1] for r in rows]}
        ).with_columns(
            embedding=pl.Series([np.asarray(r[2], dtype=np.float32) for r in rows],
                                dtype=pl.Array(pl.Float32, DIM))
        ).write_parquet(d / f"shard{si:03d}.parquet")
    return d


SPEC = [
    [("ctx1", "context", [1.0, 0.0, 0.0, 0.0]),
     ("p_x", "paper", [5.0, 0.0, 0.0, 0.0]),
     ("p_y", "paper", [0.0, 3.0, 0.0, 0.0])],
    [("ctx2", "context", [0.0, 1.0, 0.0, 0.0]),
     ("p_z", "paper", [0.0, 0.0, 7.0, 0.0]),
     ("p_w", "paper", [-2.0, 0.0, 0.0, 0.0])],
]


def _store(tmp_path):
    prefix = tmp_path / "mm"
    build(_shards(tmp_path, SPEC), prefix)
    return MmapScorer(prefix)


def test_vectors_are_normalised_at_build_time(tmp_path):
    # Query-time normalisation would touch every page on every call, which is
    # exactly the whole-matrix read the mmap exists to avoid.
    prefix = tmp_path / "mm"
    build(_shards(tmp_path, SPEC), prefix)
    mat = np.load(f"{prefix}_paper.npy", mmap_mode="r")
    assert np.allclose(np.linalg.norm(mat.astype(np.float32), axis=1), 1.0,
                       atol=1e-3)


def test_scores_are_cosines_not_dot_products_of_raw_vectors(tmp_path):
    s = _store(tmp_path)
    assert s("ctx1", "p_x") == pytest.approx(1.0, abs=1e-3)    # parallel, 5x length
    assert s("ctx1", "p_y") == pytest.approx(0.0, abs=1e-3)
    assert s("ctx1", "p_w") == pytest.approx(-1.0, abs=1e-3)   # antiparallel
    assert s("ctx2", "p_y") == pytest.approx(1.0, abs=1e-3)


def test_pool_scores_agree_with_single_calls_despite_the_sorted_gather(tmp_path):
    # pool_scores reorders rows to read the mmap in file order and must undo
    # that permutation; if it did not, every candidate would receive another
    # candidate's score.
    s = _store(tmp_path)
    cands = ("p_z", "p_x", "p_w", "p_y")      # deliberately not in row order
    pooled = s.pool_scores("ctx1", cands)
    assert pooled == {c: pytest.approx(s("ctx1", c), abs=1e-3) for c in cands}
    assert pooled["p_x"] == pytest.approx(1.0, abs=1e-3)
    assert pooled["p_w"] == pytest.approx(-1.0, abs=1e-3)


def test_keys_keep_their_rows_across_shards(tmp_path):
    s = _store(tmp_path)
    # p_z lives in the second shard; a fill-offset bug would hand it p_x's row.
    assert s("ctx1", "p_z") == pytest.approx(0.0, abs=1e-3)
    assert s.pool_scores("ctx2", ("p_z",))["p_z"] == pytest.approx(0.0, abs=1e-3)


def test_missing_key_is_refused(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(SystemExit, match="no embedding"):
        s("ctx1", "absent")
    with pytest.raises(SystemExit, match="no embedding"):
        s.pool_scores("absent", ("p_x",))


def test_zero_norm_vector_is_refused_at_build(tmp_path):
    spec = [[("ctx1", "context", [1.0, 0.0, 0.0, 0.0]),
             ("p_x", "paper", [0.0, 0.0, 0.0, 0.0])]]
    with pytest.raises(SystemExit, match="zero norm"):
        build(_shards(tmp_path, spec), tmp_path / "mm")
