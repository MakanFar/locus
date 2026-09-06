import numpy as np

from locus.scoring.retrieve import top_k


class _Store:
    """A stand-in for MmapScorer holding unit-normalised toy vectors."""

    def __init__(self, docs, ctxs):
        d = np.asarray(docs, dtype=np.float32)
        c = np.asarray(ctxs, dtype=np.float32)
        self._doc = (d / np.linalg.norm(d, axis=1, keepdims=True)).astype(
            np.float16)
        self._ctx = (c / np.linalg.norm(c, axis=1, keepdims=True)).astype(
            np.float16)
        self._doc_ix = {f"p{i}": i for i in range(len(d))}
        self._ctx_ix = {f"c{i}": i for i in range(len(c))}


def test_top_k_ranks_by_cosine():
    # p0 points with the query, p2 against it.
    store = _Store([[1, 0], [0, 1], [-1, 0]], [[1, 0]])
    got = top_k(store, ["c0"], k=3)
    assert got.tolist() == [[0, 1, 2]]


def test_block_boundaries_do_not_change_the_result():
    # The running top-k merge across paper blocks is where an off-by-one
    # silently truncates. Scanning in blocks of one must agree with scanning
    # in a single block.
    rng = np.random.default_rng(0)
    docs = rng.normal(size=(23, 5))
    ctxs = rng.normal(size=(4, 5))
    store = _Store(docs, ctxs)
    ids = ["c0", "c1", "c2", "c3"]
    whole = top_k(store, ids, k=6, paper_block=100)
    for block in (1, 2, 3, 7, 11):
        assert top_k(store, ids, k=6, paper_block=block).tolist() == \
            whole.tolist(), f"paper_block={block} disagreed"


def test_query_chunking_does_not_change_the_result():
    rng = np.random.default_rng(1)
    store = _Store(rng.normal(size=(30, 4)), rng.normal(size=(9, 4)))
    ids = [f"c{i}" for i in range(9)]
    whole = top_k(store, ids, k=5, query_chunk=100)
    for chunk in (1, 2, 4, 8):
        assert top_k(store, ids, k=5, query_chunk=chunk).tolist() == \
            whole.tolist(), f"query_chunk={chunk} disagreed"


def test_k_larger_than_the_corpus_is_clamped():
    store = _Store([[1, 0], [0, 1]], [[1, 1]])
    assert top_k(store, ["c0"], k=50).shape == (1, 2)


def test_results_are_sorted_best_first():
    # Sequential completion walks the candidate list assuming nothing about order, but
    # first-stage recall@1 reads position 0, so the ordering has to be real.
    rng = np.random.default_rng(2)
    docs = rng.normal(size=(40, 6))
    store = _Store(docs, rng.normal(size=(3, 6)))
    got = top_k(store, ["c0", "c1", "c2"], k=8, paper_block=7)
    for qi, row in enumerate(got):
        q = store._ctx[qi].astype(np.float32)
        sims = [float(store._doc[j].astype(np.float32) @ q) for j in row]
        assert sims == sorted(sims, reverse=True)


def test_merge_topk_bounds_the_working_set():
    # Dropping the trim is invisible in the OUTPUT -- the final sort still
    # returns the right k -- and turns memory from O(k) into O(corpus). So
    # the width, not the values, is what this pins.
    from locus.scoring.retrieve import merge_topk
    k = 4
    bs = np.empty((3, 0), dtype=np.float32)
    bi = np.empty((3, 0), dtype=np.int64)
    for block in range(6):
        s = np.full((3, 5), float(block), dtype=np.float32)
        i = np.arange(5, dtype=np.int64)[None, :].repeat(3, 0) + block * 5
        bs, bi = merge_topk(bs, bi, s, i, k)
        assert bs.shape[1] <= k, f"width grew to {bs.shape[1]} after {block+1}"
        assert bi.shape[1] == bs.shape[1]


def test_merge_topk_keeps_the_best_not_the_first():
    from locus.scoring.retrieve import merge_topk
    bs = np.array([[0.9, 0.1]], dtype=np.float32)
    bi = np.array([[10, 11]], dtype=np.int64)
    bs, bi = merge_topk(bs, bi, np.array([[0.5]], dtype=np.float32),
                        np.array([[12]], dtype=np.int64), k=2)
    assert sorted(bi[0].tolist()) == [10, 12]
