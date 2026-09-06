import numpy as np
import polars as pl
import pytest

from locus.scoring.dense import DenseScorer, load_embeddings

DIM = 4


def _write(tmp_path, rows):
    frame = pl.DataFrame(
        {"key": [r[0] for r in rows], "kind": [r[1] for r in rows]}
    ).with_columns(
        embedding=pl.Series(
            [np.asarray(r[2], dtype=np.float32) for r in rows],
            dtype=pl.Array(pl.Float32, DIM),
        )
    )
    path = tmp_path / "e.parquet"
    frame.write_parquet(path)
    return path


def _basic(tmp_path):
    return _write(tmp_path, [
        ("ctx1", "context", [1.0, 0.0, 0.0, 0.0]),
        ("ctx2", "context", [0.0, 1.0, 0.0, 0.0]),
        ("p_x",  "paper",   [2.0, 0.0, 0.0, 0.0]),   # parallel to ctx1
        ("p_y",  "paper",   [0.0, 5.0, 0.0, 0.0]),   # parallel to ctx2
        ("p_z",  "paper",   [-1.0, 0.0, 0.0, 0.0]),  # opposed to ctx1
    ])


def test_cosine_ignores_magnitude(tmp_path):
    # p_x is 2x the length of ctx1 but points the same way: cosine 1.0, not 2.0.
    # If normalisation were dropped this would be the score that silently
    # rewarded long abstracts.
    s = DenseScorer(_basic(tmp_path))
    assert s("ctx1", "p_x") == pytest.approx(1.0)
    assert s("ctx2", "p_y") == pytest.approx(1.0)
    assert s("ctx1", "p_z") == pytest.approx(-1.0)
    assert s("ctx1", "p_y") == pytest.approx(0.0)


def test_pool_scores_agrees_with_call(tmp_path):
    # The lambda sweep uses the batched form and the null controls use __call__;
    # a divergence between them would make the sweep optimise a different
    # quantity from the one reported.
    s = DenseScorer(_basic(tmp_path))
    cands = ("p_x", "p_y", "p_z")
    pooled = s.pool_scores("ctx1", cands)
    assert pooled == {c: pytest.approx(s("ctx1", c)) for c in cands}


def test_missing_key_is_refused_not_scored_zero(tmp_path):
    # 0.0 is a plausible cosine, so a missing vector must not become one --
    # it would tie that candidate near the pool mean and look merely weak.
    s = DenseScorer(_basic(tmp_path))
    with pytest.raises(SystemExit, match="no embedding"):
        s("ctx1", "p_absent")
    with pytest.raises(SystemExit, match="no embedding"):
        s.pool_scores("ctx_absent", ("p_x",))


def test_zero_norm_vector_is_refused(tmp_path):
    path = _write(tmp_path, [
        ("ctx1", "context", [1.0, 0.0, 0.0, 0.0]),
        ("p_x",  "paper",   [0.0, 0.0, 0.0, 0.0]),
    ])
    with pytest.raises(SystemExit, match="zero norm"):
        load_embeddings(path)


def test_missing_kind_is_refused(tmp_path):
    path = _write(tmp_path, [("ctx1", "context", [1.0, 0.0, 0.0, 0.0])])
    with pytest.raises(SystemExit, match="no rows of kind 'paper'"):
        load_embeddings(path)


def test_float16_storage_normalises_and_scores(tmp_path):
    # The corpus-wide candidate matrix is 1.48M x 768; float16 halves it to
    # 2.28 GB, which is what makes it loadable at all on a 16 GB machine.
    # Arithmetic must still happen in float32 -- fp16 dot products over 768
    # terms lose precision that fp16 *storage* does not.
    rows = [("ctx1", "context", [3.0, 4.0, 0.0, 0.0]),
            ("p_x", "paper", [6.0, 8.0, 0.0, 0.0]),   # parallel to ctx1
            ("p_y", "paper", [0.0, 0.0, 1.0, 0.0])]
    frame = pl.DataFrame(
        {"key": [r[0] for r in rows], "kind": [r[1] for r in rows]}
    ).with_columns(
        embedding=pl.Series([np.asarray(r[2], dtype=np.float16) for r in rows],
                            dtype=pl.Array(pl.Float16, DIM))
    )
    path = tmp_path / "f16.parquet"
    frame.write_parquet(path)

    parts = load_embeddings(path)
    assert parts["paper"][1].dtype == np.float16   # storage stays narrow
    s = DenseScorer(path)
    assert s("ctx1", "p_x") == pytest.approx(1.0, abs=1e-3)
    assert s("ctx1", "p_y") == pytest.approx(0.0, abs=1e-3)
    assert s.pool_scores("ctx1", ("p_x", "p_y"))["p_x"] == pytest.approx(1.0, abs=1e-3)


def test_chunked_normalisation_matches_a_single_pass(tmp_path, monkeypatch):
    # The chunk boundary must not change anything; a bug here would leave one
    # block unnormalised and silently reweight part of the corpus.
    import locus.scoring.dense as dense_mod
    rng = np.random.default_rng(0)
    v = rng.standard_normal((37, DIM)).astype(np.float32) * 10.0
    frame = pl.DataFrame(
        {"key": [f"p{i}" for i in range(37)], "kind": ["paper"] * 37}
    ).with_columns(embedding=pl.Series(list(v), dtype=pl.Array(pl.Float32, DIM)))
    frame = frame.vstack(pl.DataFrame(
        {"key": ["ctx1"], "kind": ["context"]}
    ).with_columns(embedding=pl.Series([v[0]], dtype=pl.Array(pl.Float32, DIM))))
    path = tmp_path / "c.parquet"
    frame.write_parquet(path)

    monkeypatch.setattr(dense_mod, "CHUNK", 1_000_000)
    whole = load_embeddings(path)["paper"][1].copy()
    monkeypatch.setattr(dense_mod, "CHUNK", 5)
    chunked = load_embeddings(path)["paper"][1]
    assert np.array_equal(whole, chunked)
    assert np.allclose(np.linalg.norm(chunked, axis=1), 1.0, atol=1e-6)


def test_a_directory_of_shards_loads_as_one_matrix(tmp_path):
    # Large builds are sharded on the GPU box; the loader has to stitch them
    # back in a reproducible order or every key would map to the wrong row.
    shard_dir = tmp_path / "sharded"
    shard_dir.mkdir()
    rows = [("ctx1", "context", [1.0, 0.0, 0.0, 0.0]),
            ("p_x", "paper", [1.0, 0.0, 0.0, 0.0]),
            ("p_y", "paper", [0.0, 1.0, 0.0, 0.0]),
            ("p_z", "paper", [0.0, 0.0, 1.0, 0.0])]
    for i, chunk in enumerate([rows[:2], rows[2:]]):
        pl.DataFrame(
            {"key": [r[0] for r in chunk], "kind": [r[1] for r in chunk]}
        ).with_columns(
            embedding=pl.Series([np.asarray(r[2], dtype=np.float32) for r in chunk],
                                dtype=pl.Array(pl.Float32, DIM))
        ).write_parquet(shard_dir / f"shard{i:03d}.parquet")

    s = DenseScorer(shard_dir)
    assert s("ctx1", "p_x") == pytest.approx(1.0)
    assert s("ctx1", "p_y") == pytest.approx(0.0)
    assert s("ctx1", "p_z") == pytest.approx(0.0)


def test_an_empty_shard_directory_is_refused(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match=r"no \.parquet shards"):
        load_embeddings(empty)


def test_shards_with_mixed_kinds_keep_key_to_row_alignment(tmp_path):
    # Each shard carries both kinds interleaved, and rows are scattered into
    # per-kind matrices as shards stream past. If the fill offset drifted,
    # every key would silently map to another paper's vector -- scores would
    # stay plausible and be entirely wrong.
    shard_dir = tmp_path / "mixed"
    shard_dir.mkdir()
    expected = {}
    for si in range(3):
        rows = []
        for j in range(4):
            kind = "context" if j % 2 == 0 else "paper"
            key = f"{kind}_{si}_{j}"
            vec = [float(si + 1), float(j + 1), 0.0, 0.0]
            rows.append((key, kind, vec))
            expected[key] = np.asarray(vec, dtype=np.float32)
        pl.DataFrame(
            {"key": [r[0] for r in rows], "kind": [r[1] for r in rows]}
        ).with_columns(
            embedding=pl.Series([np.asarray(r[2], dtype=np.float32) for r in rows],
                                dtype=pl.Array(pl.Float32, DIM))
        ).write_parquet(shard_dir / f"shard{si:03d}.parquet")

    parts = load_embeddings(shard_dir)
    for kind in ("context", "paper"):
        index, mat = parts[kind]
        assert len(index) == 6
        for key, row in index.items():
            want = expected[key]
            want = want / np.linalg.norm(want)
            assert np.allclose(mat[row], want, atol=1e-6), key


def test_a_variable_length_list_column_is_refused_by_name(tmp_path):
    """The mistake a bring-your-own-model user actually makes.

    Building the column from a list of vectors makes polars infer
    `List(Float32)` rather than `Array(Float32, dim)`. That used to reach
    `block.shape[1]` and raise `IndexError: tuple index out of range`, naming
    neither the column nor the cause, while README.md claimed the loader
    rejected it. The message must name the column, the dtype it got, the
    dtype it needs, and how to build it.
    """
    frame = pl.DataFrame({
        "key": ["ctx1", "p_x"],
        "kind": ["context", "paper"],
        "embedding": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    })
    assert not isinstance(frame.schema["embedding"], pl.Array), (
        "fixture is meant to produce a variable-length List column"
    )
    path = tmp_path / "list.parquet"
    frame.write_parquet(path)

    with pytest.raises(SystemExit) as exc:
        load_embeddings(path)
    message = str(exc.value)
    assert "embedding" in message
    assert "Array(Float32, dim)" in message
    assert "List" in message


def test_a_ragged_list_column_is_refused_too(tmp_path):
    """The same shape, but with no width to infer at all: a reshape could not
    rescue this one, so it must fail on the dtype rather than on whatever the
    first row happened to be."""
    frame = pl.DataFrame({
        "key": ["ctx1", "p_x"],
        "kind": ["context", "paper"],
        "embedding": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    })
    path = tmp_path / "ragged.parquet"
    frame.write_parquet(path)
    with pytest.raises(SystemExit, match="fixed-width"):
        load_embeddings(path)


def test_a_parquet_with_no_embedding_column_is_refused(tmp_path):
    frame = pl.DataFrame(
        {"key": ["ctx1", "p_x"], "kind": ["context", "paper"]}
    )
    path = tmp_path / "no_embedding.parquet"
    frame.write_parquet(path)
    with pytest.raises(SystemExit, match="no 'embedding' column"):
        load_embeddings(path)
