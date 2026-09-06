import collections
import itertools

import numpy as np
import pytest
from scipy import sparse

from locus.scoring.node2vec import (
    NodeVectors,
    _is_edge,
    edge_keys,
    walks,
)


def _graph(edges, n):
    r = [a for a, b in edges] + [b for a, b in edges]
    c = [b for a, b in edges] + [a for a, b in edges]
    m = sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    m.sort_indices()
    return m


# 0-1, 1-2, 1-3, 0-2. From t=0 at v=1 the three neighbours are exactly the
# three cases the bias distinguishes: 0 is the return, 2 is a neighbour of the
# return, 3 is neither.
BOWTIE = _graph([(0, 1), (1, 2), (1, 3), (0, 2)], 4)


def _second_steps(g, p, q, n=4000, seed=0):
    """Where does a walk starting 0 -> 1 go next?"""
    W = walks(g, p=p, q=q, num_walks=n, length=3, seed=seed)
    started = W[(W[:, 0] == 0) & (W[:, 1] == 1)]
    return collections.Counter(started[:, 2].tolist()), len(started)


class TestEdgeLookup:
    def test_is_edge_agrees_with_the_matrix(self):
        keys = edge_keys(BOWTIE)
        dense = BOWTIE.toarray() > 0
        src, dst = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
        got = _is_edge(keys, src.ravel(), dst.ravel(), 4).reshape(4, 4)
        assert (got == dense).all()

    def test_a_probe_past_the_last_key_is_not_an_edge(self):
        # searchsorted returns len(keys) for anything above the maximum; an
        # unclamped index would raise, and a clamped one must still compare
        # false rather than silently matching the last edge.
        keys = edge_keys(BOWTIE)
        assert not _is_edge(keys, np.array([3]), np.array([3]), 4)[0]


class TestWalksStayOnTheGraph:
    def test_every_consecutive_pair_is_an_edge(self):
        W = walks(BOWTIE, p=1.0, q=1.0, num_walks=50, length=8, seed=0)
        dense = BOWTIE.toarray() > 0
        for w in W:
            for a, b in itertools.pairwise(w):
                assert dense[a, b], f"{a}->{b} is not an edge"

    def test_isolated_nodes_start_no_walks(self):
        # Node 4 has no edges. Starting a walk there would emit a length-1
        # sentence of a token that has no context, which word2vec would then
        # give an untrained vector -- indistinguishable, downstream, from a
        # vector that means something.
        g = _graph([(0, 1), (1, 2), (1, 3), (0, 2)], 5)
        W = walks(g, p=1.0, q=1.0, num_walks=3, length=5, seed=0)
        assert 4 not in set(W[:, 0].tolist())
        assert W.shape[0] == 3 * 4

    def test_a_fixed_seed_reproduces_the_walks(self):
        a = walks(BOWTIE, p=1.0, q=0.5, num_walks=20, length=6, seed=7)
        b = walks(BOWTIE, p=1.0, q=0.5, num_walks=20, length=6, seed=7)
        assert (a == b).all()
        c = walks(BOWTIE, p=1.0, q=0.5, num_walks=20, length=6, seed=8)
        assert not (a == c).all()


class TestSecondOrderBias:
    """The p/q bias is what separates node2vec from DeepWalk.

    Without these, every grid point over (p, q) would train on the same
    uniform walks and the sweep would report a winning (p, q) that meant
    nothing at all.
    """

    def test_unbiased_walks_choose_uniformly(self):
        counts, n = _second_steps(BOWTIE, p=1.0, q=1.0)
        for node in (0, 2, 3):
            assert counts[node] / n == pytest.approx(1 / 3, abs=0.03), node

    def test_a_small_p_makes_the_walk_return(self):
        # 1/p = 10 on the return, 1 elsewhere -> 10/12 of steps go back to 0.
        counts, n = _second_steps(BOWTIE, p=0.1, q=1.0)
        assert counts[0] / n == pytest.approx(10 / 12, abs=0.03)

    def test_a_large_p_suppresses_the_return(self):
        counts, n = _second_steps(BOWTIE, p=10.0, q=1.0)
        assert counts[0] / n == pytest.approx(0.1 / 2.1, abs=0.03)

    def test_a_small_q_pushes_the_walk_outward(self):
        # 3 is the only neighbour of 1 that is NOT adjacent to 0, so 1/q = 10
        # applies to it alone: 10/12 of steps.
        counts, n = _second_steps(BOWTIE, p=1.0, q=0.1)
        assert counts[3] / n == pytest.approx(10 / 12, abs=0.03)

    def test_a_large_q_keeps_the_walk_local(self):
        counts, n = _second_steps(BOWTIE, p=1.0, q=10.0)
        assert counts[3] / n == pytest.approx(0.1 / 2.1, abs=0.03)


class TestNodeVectors:
    def test_similarity_is_clipped_at_zero_like_ppmi(self, tiny_vectors):
        # Without the clip a pair absent from the graph would score 0.0 and so
        # rank ABOVE a pair the embedding judged actively dissimilar, which
        # inverts what "no evidence" means in the structural term.
        nv = tiny_vectors
        assert nv.value("c", "a1") == pytest.approx(0.8, abs=1e-6)
        assert nv.value("c", "a2") == 0.0

    def test_a_paper_outside_the_graph_scores_zero(self, tiny_vectors):
        nv = tiny_vectors
        assert nv.value("c", "absent") == 0.0
        assert nv.value("absent", "a1") == 0.0
        assert nv.aggregate("c", frozenset(["absent"]), "mean") == 0.0

    def test_mean_divides_by_the_full_anchor_set(self, tiny_vectors):
        # Matches PPMI.aggregate: an anchor with no node is evidence we do not
        # have, not evidence that does not exist. Dividing by the anchors that
        # happen to have vectors would make the arms differ in coverage as
        # well as in measure.
        nv = tiny_vectors
        A = frozenset(["a1", "absent"])
        assert nv.aggregate("c", A, "mean") == pytest.approx(0.4, abs=1e-6)
        assert nv.aggregate("c", A, "masked_mean") == pytest.approx(0.8, abs=1e-6)

    def test_max_ignores_set_size(self, tiny_vectors):
        assert tiny_vectors.aggregate("c", frozenset(["a1", "a2"]), "max") == \
            pytest.approx(0.8, abs=1e-6)

    def test_mean_over_is_the_mean_aggregator(self, tiny_vectors):
        nv = tiny_vectors
        A = frozenset(["a1", "a2"])
        assert nv.mean_over("c", A) == nv.aggregate("c", A, "mean")

    def test_an_empty_anchor_set_is_zero_not_nan(self, tiny_vectors):
        # probe._band raises on NaN, so an empty set must not divide by zero.
        for how in ("mean", "masked_mean", "max"):
            assert tiny_vectors.aggregate("c", frozenset(), how) == 0.0

    def test_unknown_aggregator_is_refused(self, tiny_vectors):
        with pytest.raises(ValueError, match="unknown aggregator"):
            tiny_vectors.aggregate("c", frozenset(["a1"]), "median")


class TestAssociationFactory:
    """One factory, because three hand-copied branches is how one of sweep,
     and swap ends up reading a different graph from the other two."""

    def _graph(self):
        from locus.core.types import ContextRec
        from locus.scoring.cocitation import count_sites

        def rec(cid, refid, loc):
            return ContextRec(context_id=f"{cid}_{refid}_0", citing_id=cid,
                              refid=refid, raw="", masked="", location=loc)
        sites = [("P", [rec("P", "a", 0), rec("P", "b", 0), rec("P", "c", 1)])]
        return count_sites(sites)

    def test_each_name_builds_a_different_measure(self):
        from locus.scoring.node2vec import association
        g = self._graph()
        p = association("ppmi", g, 0.75, 1, None)
        c = association("count", g, None, 1, None)
        assert p.value("a", "b") != c.value("a", "b")

    def test_an_unknown_name_is_refused_rather_than_defaulted(self):
        # Silently falling back to PPMI would report a node2vec arm that had
        # never run node2vec, and every number in it would look plausible.
        from locus.scoring.node2vec import association
        with pytest.raises(ValueError, match="unknown association measure"):
            association("word2vec", self._graph(), 0.75, 1, None)

    def test_node2vec_loads_vectors_and_ignores_the_smoothing_axes(self, tmp_path):
        import json as _json

        from locus.scoring.node2vec import association
        v = np.array([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32)
        np.save(tmp_path / "nv_vectors.npy", v)
        (tmp_path / "nv_index.json").write_text(_json.dumps({"a": 0, "b": 1}))
        a = association("node2vec", self._graph(), 0.5, 1, tmp_path / "nv")
        b = association("node2vec", self._graph(), 1.0, 99, tmp_path / "nv")
        assert a.value("a", "b") == b.value("a", "b") == pytest.approx(0.6)


class TestCorpus:
    def test_every_walk_becomes_one_line_of_paper_ids(self, tmp_path):
        from locus.scoring.node2vec import write_corpus
        W = np.array([[0, 1, 2], [2, 0, 0]], dtype=np.int32)
        p = tmp_path / "w.txt"
        assert write_corpus(W, ("pA", "pB", "pC"), p, chunk=1) == 2
        lines = p.read_text().splitlines()
        assert lines == ["pA pB pC", "pC pA pA"]

    def test_chunking_does_not_change_the_file(self, tmp_path):
        # The chunked write exists to bound memory; a boundary bug there would
        # drop or duplicate walks and only show up as a slightly worse
        # embedding, which is invisible.
        from locus.scoring.node2vec import write_corpus
        rng = np.random.default_rng(0)
        W = rng.integers(0, 5, size=(37, 6)).astype(np.int32)
        vocab = tuple(f"p{i}" for i in range(5))
        ref = None
        for c in (1, 2, 7, 37, 1000):
            p = tmp_path / f"w{c}.txt"
            write_corpus(W, vocab, p, chunk=c)
            got = p.read_text()
            assert len(got.splitlines()) == 37, c
            if ref is None:
                ref = got
            assert got == ref, f"chunk={c} differed"


class TestIncrementalAccumulation:
    def _nv(self):
        v = np.array([[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0], [0.6, 0.8]],
                     dtype=np.float32)
        return NodeVectors(vectors=v,
                           index={"c": 0, "a1": 1, "a2": 2, "d": 3})

    def test_accumulating_anchors_matches_the_one_shot_mean(self):
        # The two paths must agree, or sequential completion and LOCUS-Rank would be
        # scoring different things under the same name.
        nv = self._nv()
        cands = ["c", "d"]
        prep = nv.prepare(cands)
        sums = np.zeros(len(cands))
        anchors = ["a1", "a2"]
        for a in anchors:
            nv.add_to(sums, a, prep)
        got = sums / len(anchors)
        want = [nv.aggregate(c, frozenset(anchors), "mean") for c in cands]
        assert got == pytest.approx(want)

    def test_negative_similarity_is_clipped_in_the_incremental_path_too(self):
        # a2 points away from c. Without the clip the running sum would go
        # negative here and positive in `aggregate`, and only the sequential
        # experiment would show it.
        nv = self._nv()
        prep = nv.prepare(["c"])
        sums = np.zeros(1)
        nv.add_to(sums, "a2", prep)
        assert sums[0] == 0.0

    def test_an_unknown_candidate_gets_a_zero_row_not_a_dropped_column(self):
        nv = self._nv()
        prep = nv.prepare(["c", "absent"])
        assert prep.shape == (2, 2)
        assert (prep[1] == 0).all()
        sums = np.zeros(2)
        nv.add_to(sums, "a1", prep)
        assert sums[1] == 0.0

    def test_an_out_of_vocab_anchor_contributes_nothing(self):
        nv = self._nv()
        sums = np.zeros(1)
        nv.add_to(sums, "not-in-graph", nv.prepare(["c"]))
        assert sums[0] == 0.0
