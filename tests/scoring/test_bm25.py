import math

import polars as pl
import pytest

from locus.scoring.bm25 import BM25Scorer


def _write(tmp_path, rows, name="t.parquet"):
    path = tmp_path / name
    pl.DataFrame(
        {"key": [r[0] for r in rows], "kind": [r[1] for r in rows],
         "text": [r[2] for r in rows]}
    ).write_parquet(path)
    return path


def _basic(tmp_path):
    return _write(tmp_path, [
        ("ctx1", "context", "graph neural network"),
        ("ctx2", "context", "protein folding"),
        ("p_hit",  "paper", "Graph neural network methods for citation graphs"),
        ("p_miss", "paper", "Protein folding with deep learning"),
        ("p_none", "paper", "Quantum chromodynamics on the lattice"),
    ])


def test_scores_the_lexically_matching_candidate_highest(tmp_path):
    s = BM25Scorer(_basic(tmp_path))
    pool = s.pool_scores("ctx1", ["p_hit", "p_miss", "p_none"])
    assert max(pool, key=pool.get) == "p_hit"


def test_a_candidate_sharing_no_term_scores_exactly_zero(tmp_path):
    # Not "small" -- zero. Every BM25 term is gated on tf > 0, so a document
    # with no query term contributes no summand at all. A pool of these ties,
    # which is what lets the harness's tie rule apply rather than an arbitrary
    # ordering of floating-point noise.
    s = BM25Scorer(_basic(tmp_path))
    assert s("ctx2", "p_none") == 0.0


def test_call_and_pool_scores_agree(tmp_path):
    s = BM25Scorer(_basic(tmp_path))
    cands = ["p_hit", "p_miss", "p_none"]
    assert s.pool_scores("ctx1", cands) == {c: s("ctx1", c) for c in cands}


def test_idf_is_never_negative_for_a_term_in_every_document(tmp_path):
    # The textbook Robertson/Sparck-Jones idf, log((N-n+0.5)/(n+0.5)), goes
    # NEGATIVE once a term appears in more than half the corpus, so a document
    # can be penalised for containing a query term. The +1 inside the log is
    # what forbids that. A negative idf would make a pool's ordering depend on
    # corpus frequency in the wrong direction.
    path = _write(tmp_path, [
        ("c", "context", "ubiquitous"),
        ("p1", "paper", "ubiquitous term here"),
        ("p2", "paper", "ubiquitous term there"),
        ("p3", "paper", "ubiquitous term everywhere"),
    ])
    s = BM25Scorer(path)
    assert all(s("c", p) > 0.0 for p in ("p1", "p2", "p3"))


def test_length_normalisation_prefers_the_shorter_document(tmp_path):
    # Same single occurrence of the query term; b=0.75 must break the tie
    # toward the shorter document. With b=0 it would not.
    path = _write(tmp_path, [
        ("c", "context", "signal"),
        ("short", "paper", "signal"),
        ("long", "paper", "signal " + " ".join(f"w{i}" for i in range(200))),
    ])
    s = BM25Scorer(path)
    assert s("c", "short") > s("c", "long")
    flat = BM25Scorer(path, b=0.0)
    assert flat("c", "short") == pytest.approx(flat("c", "long"))


def test_saturation_is_sublinear_in_term_frequency(tmp_path):
    # BM25's defining property against raw tf: ten occurrences must score less
    # than ten times one occurrence, or the measure is just weighted tf.
    path = _write(tmp_path, [
        ("c", "context", "signal"),
        ("one", "paper", "signal " + " ".join(f"w{i}" for i in range(9))),
        ("ten", "paper", " ".join(["signal"] * 10)),
    ])
    s = BM25Scorer(path, b=0.0)
    assert s("c", "one") < s("c", "ten") < 10 * s("c", "one")


def test_scores_are_finite_for_an_empty_query(tmp_path):
    # probe._band raises on any NaN in a pool, so an empty or all-unknown
    # query must tie the pool at 0.0 rather than produce nan.
    path = _write(tmp_path, [
        ("c", "context", ""),
        ("p1", "paper", "anything"),
        ("p2", "paper", "something else"),
    ])
    s = BM25Scorer(path)
    pool = s.pool_scores("c", ["p1", "p2"])
    assert set(pool.values()) == {0.0}
    assert all(math.isfinite(v) for v in pool.values())


def test_unknown_context_is_refused_by_name(tmp_path):
    s = BM25Scorer(_basic(tmp_path))
    with pytest.raises(KeyError, match="nope"):
        s.pool_scores("nope", ["p_hit"])


def test_unknown_candidate_is_refused_by_name(tmp_path):
    s = BM25Scorer(_basic(tmp_path))
    with pytest.raises(KeyError, match="ghost"):
        s.pool_scores("ctx1", ["p_hit", "ghost"])


def test_idf_is_built_from_papers_not_contexts(tmp_path):
    # The candidate universe is the paper side of the file. If contexts leaked
    # into the document frequency count, N and every n would change and the
    # scores would silently depend on how many contexts the split happens to
    # carry -- a number that has nothing to do with the candidates.
    rows = [("c", "context", "alpha"), ("p1", "paper", "alpha"), ("p2", "paper", "beta")]
    base = BM25Scorer(_write(tmp_path, rows, "a.parquet"))("c", "p1")
    padded = rows + [(f"x{i}", "context", "alpha") for i in range(50)]
    assert BM25Scorer(_write(tmp_path, padded, "b.parquet"))("c", "p1") == base


def test_document_frequency_counts_documents_not_occurrences(tmp_path):
    # `Counter.update(Counter)` adds frequencies rather than marking presence.
    # A term repeated inside one document must not raise its df: here "signal"
    # is in exactly one of two papers either way, so both files must agree.
    once = _write(tmp_path, [
        ("c", "context", "signal"),
        ("p1", "paper", "signal filler filler"),
        ("p2", "paper", "unrelated words here"),
    ], "once.parquet")
    many = _write(tmp_path, [
        ("c", "context", "signal"),
        ("p1", "paper", "signal signal signal"),
        ("p2", "paper", "unrelated words here"),
    ], "many.parquet")
    assert BM25Scorer(once)._idf["signal"] == BM25Scorer(many)._idf["signal"]
