
import itertools
import math

import numpy as np
import pytest
from scipy import sparse

from locus.core.types import ContextRec
from locus.scoring import cocitation
from locus.scoring.cocitation import PPMI, CoGraph, count_sites


def _rec(cid, refid, loc, ctx=None):
    return ContextRec(
        context_id=ctx or f"{cid}_{refid}_{loc}",
        citing_id=cid,
        refid=refid,
        raw="",
        masked="",
        location=loc,
    )


def _c(g, a, b):
    return int(g.counts[g.index[a], g.index[b]])


class TestCountSites:
    def test_a_site_of_three_contributes_all_three_pairs(self):
        recs = [_rec("P", "A", 0), _rec("P", "B", 0), _rec("P", "C", 0)]
        g = count_sites([("P", recs)])
        assert _c(g, "A", "B") == _c(g, "A", "C") == _c(g, "B", "C") == 1

    def test_counts_are_symmetric(self):
        g = count_sites([("P", [_rec("P", "A", 0), _rec("P", "B", 0)])])
        assert _c(g, "A", "B") == _c(g, "B", "A") == 1

    def test_a_singleton_site_contributes_nothing(self):
        g = count_sites([("P", [_rec("P", "A", 0)])])
        assert g.counts.nnz == 0
        assert g.n_total == 0

    def test_pairs_are_counted_once_per_site_not_once_per_context(self):
        # A and B share location 0, and A is cited twice at that location.
        # The spec's rule is one site = one (citing paper, location), each
        # unordered pair counted once -- so this is 1, not 2.
        recs = [
            _rec("P", "A", 0, "c1"),
            _rec("P", "A", 0, "c2"),
            _rec("P", "B", 0, "c3"),
        ]
        g = count_sites([("P", recs)])
        assert _c(g, "A", "B") == 1

    def test_separate_locations_are_separate_sites(self):
        recs = [
            _rec("P", "A", 0), _rec("P", "B", 0),
            _rec("P", "A", 1), _rec("P", "B", 1),
        ]
        assert _c(count_sites([("P", recs)]), "A", "B") == 2

    def test_separate_papers_accumulate(self):
        p1 = [_rec("P1", "A", 0), _rec("P1", "B", 0)]
        p2 = [_rec("P2", "A", 0), _rec("P2", "B", 0)]
        assert _c(count_sites([("P1", p1), ("P2", p2)]), "A", "B") == 2

    def test_no_self_pairs(self):
        g = count_sites([("P", [_rec("P", "A", 0), _rec("P", "B", 0)])])
        assert _c(g, "A", "A") == 0

    def test_marginals_and_total_are_the_ordered_row_sums(self):
        # A-B, A-C, B-C, each once, symmetric -> every marginal is 2,
        # n_total is 6 ordered entries.
        recs = [_rec("P", "A", 0), _rec("P", "B", 0), _rec("P", "C", 0)]
        g = count_sites([("P", recs)])
        assert sorted(g.marginals.tolist()) == [2, 2, 2]
        assert g.n_total == 6
        assert g.n_total == int(g.marginals.sum())

    def test_vocab_is_sorted_and_indexes_itself(self):
        recs = [_rec("P", "zeta", 0), _rec("P", "alpha", 0), _rec("P", "mid", 0)]
        g = count_sites([("P", recs)])
        assert list(g.vocab) == sorted(g.vocab)
        assert all(g.vocab[g.index[r]] == r for r in g.vocab)

    def test_is_independent_of_paper_arrival_order(self):
        p1 = ("P1", [_rec("P1", "B", 0), _rec("P1", "A", 0)])
        p2 = ("P2", [_rec("P2", "C", 0), _rec("P2", "A", 0)])
        g1, g2 = count_sites([p1, p2]), count_sites([p2, p1])
        assert list(g1.vocab) == list(g2.vocab)
        assert (g1.counts != g2.counts).nnz == 0

    def test_consumes_a_generator_not_just_a_list(self):
        # The train path passes iter_papers directly; a two-pass implementation
        # would silently see nothing on the second pass.
        recs = [_rec("P", "A", 0), _rec("P", "B", 0)]
        g = count_sites(iter([("P", recs)]))
        assert _c(g, "A", "B") == 1

    def test_counts_survive_the_sorted_vocab_remap(self):
        # First-appearance order (zeta, alpha, mid) differs from sorted order
        # (alpha, mid, zeta). The permutation must map first-appearance onto
        # sorted vocab correctly. Pairs have asymmetric counts to discriminate
        # permutation direction: zeta-alpha=1, alpha-mid=2, zeta-mid=0.
        p1_recs = [_rec("P1", "zeta", 0), _rec("P1", "alpha", 0)]
        p2_recs = [_rec("P2", "alpha", 0), _rec("P2", "mid", 0)]
        p3_recs = [_rec("P3", "alpha", 0), _rec("P3", "mid", 0)]
        g = count_sites([("P1", p1_recs), ("P2", p2_recs), ("P3", p3_recs)])
        assert list(g.vocab) == ["alpha", "mid", "zeta"]
        assert _c(g, "zeta", "alpha") == 1
        assert _c(g, "alpha", "mid") == 2
        assert _c(g, "zeta", "mid") == 0

    def test_bad_itemsize_raises_systemexit_before_reading_any_papers(self, monkeypatch):
        # assert is stripped under python -O and guards a real correctness
        # assumption (np.frombuffer(..., np.int32) would silently misread an
        # 8-byte array). It must also fire right after the buffers are
        # created, not after a whole streaming pass has filled them.
        class FakeArray:
            itemsize = 8

            def append(self, x):
                pass

        monkeypatch.setattr(cocitation.array, "array", lambda typecode: FakeArray())

        def boom():
            raise AssertionError(
                "count_sites read papers before checking rows.itemsize"
            )
            yield  # pragma: no cover

        with pytest.raises(SystemExit):
            count_sites(boom())

    def test_marker_anchors_are_not_edges(self):
        # Marker anchors must not contribute edges; only targets in gold_sets
        # do. Verify this by adding a record with marker_anchors pointing to
        # a refid that never appears as a regular target.
        recs = [
            _rec("P", "A", 0),
            _rec("P", "B", 0),
            ContextRec(
                context_id="with_marker",
                citing_id="P",
                refid="B",
                raw="",
                masked="",
                location=0,
                marker_anchors=frozenset({"C"}),
            ),
        ]
        g = count_sites([("P", recs)])
        # Only A and B should be in vocab; C should not
        assert "A" in g.vocab
        assert "B" in g.vocab
        assert "C" not in g.vocab
        assert len(g.vocab) == 2
        # A-B pair should exist once
        assert _c(g, "A", "B") == 1


class TestSaveLoad:
    def test_round_trips(self, tmp_path):
        recs = [_rec("P", "A", 0), _rec("P", "B", 0), _rec("P", "C", 0)]
        g = count_sites([("P", recs)])
        p = tmp_path / "g.npz"
        g.save(p)
        back = CoGraph.load(p)
        assert list(back.vocab) == list(g.vocab)
        assert back.n_total == g.n_total
        assert back.marginals.tolist() == g.marginals.tolist()
        assert (back.counts != g.counts).nnz == 0
        assert back.index == g.index

    def test_two_builds_produce_identical_arrays(self, tmp_path):
        # npz is a zip and stores mtimes, so the FILES are not byte-identical
        # across runs. The arrays must be.
        recs = [_rec("P", "A", 0), _rec("P", "B", 0), _rec("P", "C", 0)]
        a, b = count_sites([("P", recs)]), count_sites([("P", recs)])
        assert a.counts.indptr.tobytes() == b.counts.indptr.tobytes()
        assert a.counts.indices.tobytes() == b.counts.indices.tobytes()
        assert a.counts.data.tobytes() == b.counts.data.tobytes()
        assert a.marginals.tobytes() == b.marginals.tobytes()
        assert list(a.vocab) == list(b.vocab)


def _triangle_plus_pair():
    # A-B-C mutually co-cited at one site; A-D at another. Small enough to
    # check the arithmetic by hand.
    return count_sites([
        ("P1", [_rec("P1", "A", 0), _rec("P1", "B", 0), _rec("P1", "C", 0)]),
        ("P2", [_rec("P2", "A", 0), _rec("P2", "D", 0)]),
    ])


class TestPPMI:
    def test_alpha_one_reproduces_the_textbook_formula(self):
        # Not a remark: this is how we know the smoothing generalises the
        # textbook PPMI rather than replacing it.
        g = _triangle_plus_pair()
        p = g.ppmi(alpha=1.0, min_count=1)
        N, C = g.n_total, g.marginals
        for a in g.vocab:
            for b in g.vocab:
                i, j = g.index[a], g.index[b]
                cab = int(g.counts[i, j])
                expect = 0.0 if cab == 0 else max(
                    0.0, np.log(cab * N / (C[i] * C[j]))
                )
                assert p.value(a, b) == pytest.approx(expect)

    def test_min_count_zeroes_rare_pairs(self):
        g = _triangle_plus_pair()          # every pair occurs exactly once
        assert g.ppmi(alpha=1.0, min_count=2).matrix.nnz == 0
        assert g.ppmi(alpha=1.0, min_count=1).matrix.nnz > 0

    def test_negative_pmi_is_clipped_to_zero(self):
        # A and B are each co-cited with five other things but with each other
        # only once, so PMI(A,B) = log(1*22/(6*6)) < 0 and must not be stored.
        sites = [("P0", [_rec("P0", "A", 0), _rec("P0", "B", 0)])]
        sites += [
            (f"PX{i}", [_rec(f"PX{i}", "A", 0), _rec(f"PX{i}", f"X{i}", 0)])
            for i in range(5)
        ]
        sites += [
            (f"PY{i}", [_rec(f"PY{i}", "B", 0), _rec(f"PY{i}", f"Y{i}", 0)])
            for i in range(5)
        ]
        g = count_sites(sites)
        assert int(g.counts[g.index["A"], g.index["B"]]) == 1
        p = g.ppmi(alpha=1.0, min_count=1)
        assert p.value("A", "B") == 0.0
        assert (p.matrix.data > 0).all()

    def test_is_symmetric(self):
        g = _triangle_plus_pair()
        p = g.ppmi(alpha=0.75, min_count=1)
        for a in g.vocab:
            for b in g.vocab:
                assert p.value(a, b) == pytest.approx(p.value(b, a))

    def test_smoothing_damps_the_rare_partner_advantage(self):
        # D appears once, B twice. Raw PMI over-rewards the rarer partner;
        # alpha<1 flattens the marginal and shrinks that gap.
        g = count_sites([
            ("P1", [_rec("P1", "A", 0), _rec("P1", "B", 0)]),
            ("P2", [_rec("P2", "A", 0), _rec("P2", "B", 0)]),
            ("P3", [_rec("P3", "B", 0), _rec("P3", "E", 0)]),
            ("P4", [_rec("P4", "A", 0), _rec("P4", "D", 0)]),
        ])
        raw = g.ppmi(alpha=1.0, min_count=1)
        sm = g.ppmi(alpha=0.75, min_count=1)
        gap_raw = raw.value("A", "D") - raw.value("A", "B")
        gap_sm = sm.value("A", "D") - sm.value("A", "B")
        assert gap_sm < gap_raw

    def test_unknown_refids_score_zero_rather_than_raising(self):
        # The common case at scoring time: a test paper cites work the train
        # split never co-cited.
        p = _triangle_plus_pair().ppmi(alpha=1.0, min_count=1)
        assert p.value("A", "NEVER-SEEN") == 0.0
        assert p.value("NEVER-SEEN", "A") == 0.0


class TestMeanOver:
    def test_averages_over_every_anchor_including_unknown_ones(self):
        # The spec's term is (1/|A|) * sum over A. An anchor outside the train
        # vocabulary contributes 0 to the sum but still counts in |A|.
        p = _triangle_plus_pair().ppmi(alpha=1.0, min_count=1)
        one = p.value("A", "B")
        assert p.mean_over("A", frozenset({"B"})) == pytest.approx(one)
        assert p.mean_over("A", frozenset({"B", "NEVER-SEEN"})) == pytest.approx(one / 2)

    def test_empty_anchor_set_is_zero_not_nan(self):
        p = _triangle_plus_pair().ppmi(alpha=1.0, min_count=1)
        assert p.mean_over("A", frozenset()) == 0.0

    def test_unknown_candidate_is_zero(self):
        p = _triangle_plus_pair().ppmi(alpha=1.0, min_count=1)
        assert p.mean_over("NEVER-SEEN", frozenset({"A", "B"})) == 0.0

    def test_a_candidate_cannot_reward_itself(self):
        # A distractor may itself appear in A(l), since distractors come from
        # other locations and a marker anchor may resolve to one. Self-pairs
        # are never counted, so C(c,c)=0 and the self term is 0.
        p = _triangle_plus_pair().ppmi(alpha=1.0, min_count=1)
        assert p.value("A", "A") == 0.0
        assert p.mean_over("A", frozenset({"A"})) == 0.0


def _colliding_construction_orders():
    """Find three anchor names and two constructions of the same frozenset
    that, under THIS process's (hash-randomised) string hashing, iterate in
    different orders -- and choose PPMI-like values for them whose naive
    left-to-right sum genuinely differs between those two orders.

    This makes the discriminating property (mean_over must not depend on how
    the anchors frozenset happened to be built) reproducible regardless of
    PYTHONHASHSEED, instead of hardcoding names that only collide under one
    specific seed.
    """
    pool = [f"anchor{i}" for i in range(200)]
    raw_options = list(itertools.permutations([1e16, 1.0, -1e16]))
    for combo in itertools.combinations(pool, 3):
        seen: dict[tuple, tuple] = {}
        for order in itertools.permutations(combo):
            seen.setdefault(tuple(frozenset(order)), order)
        if len(seen) < 2:
            continue
        for io_a, io_b in itertools.combinations(seen, 2):
            for vals in raw_options:
                val_of = dict(zip(combo, vals, strict=True))

                def naive(order, val_of=val_of):
                    t = 0.0
                    for a in order:
                        t += val_of[a]
                    return t

                if naive(io_a) != naive(io_b):
                    return combo, seen[io_a], seen[io_b], val_of
    raise RuntimeError(
        "could not find a hash-collision case in this process; "
        "widen the anchor name pool"
    )


class TestMeanOverIsOrderIndependent:
    def test_construction_order_does_not_change_the_result(self):
        # Regression for the frozenset-iteration-order bug: mean_over used to
        # accumulate with a plain `+=` in whatever order the anchors
        # frozenset happened to iterate, which depends on PYTHONHASHSEED and
        # on the order the frozenset was built in -- not just on membership.
        members, order_a, order_b, val_of = _colliding_construction_orders()
        cols = {m: i + 1 for i, m in enumerate(sorted(members))}
        n = len(members) + 1
        col_ids = [cols[m] for m in members]
        data = [val_of[m] for m in members]
        matrix = sparse.csr_matrix(
            (data, ([0] * len(members), col_ids)), shape=(n, n)
        )
        index = {"C": 0, **cols}
        p = PPMI(matrix, index)

        anchors_a = frozenset(order_a)
        anchors_b = frozenset(order_b)
        # Sanity: the two frozensets really do iterate differently here --
        # otherwise this test would not be exercising the bug at all.
        assert list(anchors_a) != list(anchors_b)

        result_a = p.mean_over("C", anchors_a)
        result_b = p.mean_over("C", anchors_b)
        assert result_a == result_b

        expected = math.fsum(
            val_of[m] for m in sorted(members, key=cols.get)
        ) / len(members)
        assert result_a == expected
        assert result_b == expected


def _graph_of(sites):
    """count_sites over [(citing_id, location, [refids]), ...]."""
    by_paper = {}
    for cid, loc, refids in sites:
        by_paper.setdefault(cid, []).extend(
            _rec(cid, r, loc) for r in refids)
    return count_sites(sorted(by_paper.items()))


def test_raw_counts_returns_the_counts_not_an_association(tiny_graph):
    # The ablation is only meaningful if this really is the raw co-count. A
    # raw_counts that quietly reused the PPMI path would return log-scale
    # values and the "is it the PMI transform?" question would go unanswered.
    raw = tiny_graph.raw_counts(min_count=1)
    assert raw.value("a", "b") == pytest.approx(2.0)
    assert raw.value("a", "c") == pytest.approx(1.0)
    assert raw.value("b", "c") == pytest.approx(0.0)


def test_raw_counts_honours_min_count(tiny_graph):
    raw = tiny_graph.raw_counts(min_count=2)
    assert raw.value("a", "b") == pytest.approx(2.0)
    assert raw.value("a", "c") == pytest.approx(0.0)


def test_raw_counts_and_ppmi_disagree_on_ordering():
    # The whole point of the ablation. 'a' co-occurs with the very frequent
    # 'hub' more often in raw counts, but PPMI discounts the hub's marginal
    # and prefers the rarer, more specific partner.
    sites = [("p1", 0, ["a", "hub"]), ("p2", 0, ["a", "hub"]),
             ("p3", 0, ["a", "rare"])]
    sites += [(f"h{i}", 0, ["hub", f"x{i}"]) for i in range(40)]
    graph = _graph_of(sites)
    raw = graph.raw_counts(min_count=1)
    ppmi = graph.ppmi(alpha=1.0, min_count=1)
    assert raw.value("a", "hub") > raw.value("a", "rare")
    assert ppmi.value("a", "hub") < ppmi.value("a", "rare")


def _ppmi_of(pairs, vocab):
    """A PPMI view with hand-set values, for testing the aggregators."""
    index = {w: i for i, w in enumerate(vocab)}
    m = sparse.lil_matrix((len(vocab), len(vocab)))
    for (a, b), v in pairs.items():
        m[index[a], index[b]] = v
        m[index[b], index[a]] = v
    return PPMI(matrix=m.tocsr(), index=index)


class TestAggregators:
    def _p(self):
        # c co-cited with a1 (weak), a2 (strong); a3 has no relation at all.
        return _ppmi_of({("c", "a1"): 2.0, ("c", "a2"): 8.0},
                        ["c", "a1", "a2", "a3", "d"])

    def test_mean_is_bit_identical_to_mean_over(self):
        # The ablation only measures the aggregator if its `mean` arm
        # reproduces the published path exactly. A last-ULP difference here
        # would silently become part of the reported delta.
        ppmi = self._p()
        for anchors in (frozenset(), frozenset(["a1"]), frozenset(["a1", "a3"]),
                        frozenset(["a1", "a2", "a3"])):
            assert ppmi.aggregate("c", anchors, "mean") == \
                ppmi.mean_over("c", anchors)

    def test_masked_mean_excludes_zero_anchors_from_the_denominator(self):
        # This is the whole point: a3 contributes nothing to the numerator and
        # under `mean` still counts in the denominator, which is the dilution
        # the leave-one-out analysis measures as negative influence.
        ppmi = self._p()
        A = frozenset(["a1", "a2", "a3"])
        assert ppmi.aggregate("c", A, "mean") == pytest.approx(10.0 / 3)
        assert ppmi.aggregate("c", A, "masked_mean") == pytest.approx(10.0 / 2)

    def test_adding_a_zero_anchor_cannot_change_masked_mean_or_max(self):
        ppmi = self._p()
        small, large = frozenset(["a1", "a2"]), frozenset(["a1", "a2", "a3"])
        for how in ("masked_mean", "max"):
            assert ppmi.aggregate("c", small, how) == \
                ppmi.aggregate("c", large, how), how
        assert ppmi.aggregate("c", small, "mean") != \
            ppmi.aggregate("c", large, "mean")

    def test_max_takes_the_strongest_anchor(self):
        assert self._p().aggregate("c", frozenset(["a1", "a2", "a3"]), "max") \
            == pytest.approx(8.0)

    def test_no_evidence_is_zero_under_every_aggregator(self):
        # probe._band raises on NaN, so an all-zero anchor set must not divide
        # by an empty count.
        ppmi = self._p()
        for how in ("mean", "masked_mean", "max"):
            assert ppmi.aggregate("c", frozenset(["a3"]), how) == 0.0
            assert ppmi.aggregate("c", frozenset(), how) == 0.0
            assert ppmi.aggregate("d", frozenset(["a1"]), how) == 0.0

    def test_unknown_aggregator_is_refused(self):
        with pytest.raises(ValueError, match="unknown aggregator"):
            self._p().aggregate("c", frozenset(["a1"]), "median")


class TestIncrementalAccumulation:
    """`prepare`/`add_to` is sequential completion's inner loop; it must agree with the
    one-shot `aggregate` it replaces, or the sequential result would differ
    from the ranked one for reasons that have nothing to do with the setting."""

    def _p(self):
        return _ppmi_of({("c", "a1"): 2.0, ("c", "a2"): 8.0,
                         ("d", "a1"): 5.0}, ["c", "d", "a1", "a2", "a3"])

    def test_accumulating_anchors_matches_the_one_shot_mean(self):
        import numpy as np
        ppmi = self._p()
        cands = ["c", "d", "a3"]
        prep = ppmi.prepare(cands)
        sums = np.zeros(len(cands))
        anchors = ["a1", "a2"]
        for a in anchors:
            ppmi.add_to(sums, a, prep)
        got = sums / len(anchors)
        want = [ppmi.aggregate(c, frozenset(anchors), "mean") for c in cands]
        assert got == pytest.approx(want)

    def test_an_out_of_vocab_anchor_contributes_nothing(self):
        import numpy as np
        ppmi = self._p()
        prep = ppmi.prepare(["c", "d"])
        sums = np.zeros(2)
        ppmi.add_to(sums, "not-in-graph", prep)
        assert (sums == 0).all()

    def test_an_out_of_vocab_candidate_is_never_written(self):
        import numpy as np
        ppmi = self._p()
        cands = ["c", "absent"]
        prep = ppmi.prepare(cands)
        sums = np.zeros(2)
        ppmi.add_to(sums, "a1", prep)
        assert sums[1] == 0.0
        assert sums[0] == pytest.approx(2.0)

    def test_a_repeated_candidate_receives_the_value_at_every_position(self):
        # Candidate lists come from a prefetch and are not guaranteed unique;
        # a dict keyed by column that kept only the last position would leave
        # one copy unscored and silently break the argmax.
        import numpy as np
        ppmi = self._p()
        prep = ppmi.prepare(["c", "d", "c"])
        sums = np.zeros(3)
        ppmi.add_to(sums, "a1", prep)
        assert sums[0] == sums[2] == pytest.approx(2.0)
