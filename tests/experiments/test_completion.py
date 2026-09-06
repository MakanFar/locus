import numpy as np
import pytest
from scipy import sparse

from locus.experiments.completion import KS, revelation_order, run_location
from locus.scoring.cocitation import PPMI


class _Scorer:
    def __init__(self, table):
        self.table = table

    def pool_scores(self, ctx, candidates):
        return {c: self.table.get(c, 0.0) for c in candidates}


class _Graph:
    vocab = ("r1", "r2", "r3", "r4")


def _ppmi(pairs, vocab):
    index = {w: i for i, w in enumerate(vocab)}
    n = len(vocab)
    m = sparse.lil_matrix((n, n))
    for (a, b), v in pairs.items():
        m[index[a], index[b]] = v
        m[index[b], index[a]] = v
    return PPMI(matrix=m.tocsr(), index=index)


def _bins(vocab):
    return ({w: 0 for w in vocab}, [list(vocab)], [0])


LOC = {"citing_id": "p1", "location": 0, "gold": ["g1", "g2", "g3"],
       "candidates": ["g1", "g2", "g3", "d1", "d2", "d3"],
       "contexts": ["p1_g1_0"]}


def test_revelation_order_is_seeded_and_varies_by_seed():
    # Marker order correlates with sentence position, so the protocol must
    # not use it; the shuffle has to be reproducible but seed-dependent.
    a = revelation_order(["x", "y", "z"], 0, "k")
    assert a == revelation_order(["x", "y", "z"], 0, "k")
    assert sorted(a) == ["x", "y", "z"]
    assert any(revelation_order(["x", "y", "z"], s, "k") != a for s in (1, 2, 3))


def test_semantic_arm_ignores_anchors_entirely():
    # Cosine puts the two distractors on top. The semantic arm has no way to
    # use the seed, so it must keep proposing them regardless of PPMI.
    scorer = _Scorer({"d1": 0.9, "d2": 0.8, "g1": 0.5, "g2": 0.1, "g3": 0.1})
    ppmi = _ppmi({("g1", "g2"): 9.0, ("g1", "g3"): 9.0},
                 ["g1", "g2", "g3", "d1", "d2", "d3"])
    got = run_location(LOC, scorer, ppmi, _Graph(), _bins(_Graph().vocab),
                       lam=1.0, seed=0, arm="semantic")
    # seed counts as found, so R@1 is at least 1/3 whatever it proposes.
    assert got[1] == pytest.approx(1 / 3)


def test_strong_cocitation_lets_the_hybrid_beat_the_semantic_arm():
    # Same scores, but the two unfound golds co-occur strongly with whichever
    # gold is revealed, so the hybrid should surface them ahead of the
    # distractors while the semantic arm cannot.
    scorer = _Scorer({"d1": 0.9, "d2": 0.8, "g1": 0.5, "g2": 0.1, "g3": 0.1})
    vocab = ["g1", "g2", "g3", "d1", "d2", "d3"]
    ppmi = _ppmi({("g1", "g2"): 9.0, ("g1", "g3"): 9.0, ("g2", "g3"): 9.0}, vocab)
    args = (scorer, ppmi, _Graph(), _bins(vocab))
    sem = run_location(LOC, *args, lam=1.0, seed=0, arm="semantic")
    hyb = run_location(LOC, *args, lam=1.0, seed=0, arm="hybrid_retrieved")
    assert hyb[2] > sem[2]
    assert hyb[2] == pytest.approx(1.0)


def test_the_seed_is_never_proposed_again():
    # Give the seed the single highest cosine. If it were proposable, step 1
    # would spend itself re-proposing a citation already handed over and R@1
    # would stall at the seed alone; blocking it means step 1 buys a new gold.
    seed_paper = revelation_order(LOC["gold"], 0, "p1:0")[0]
    other_gold = next(g for g in LOC["gold"] if g != seed_paper)
    table = dict.fromkeys(LOC["candidates"], 0.1)
    table[seed_paper] = 0.99
    table[other_gold] = 0.98
    vocab = LOC["candidates"]
    got = run_location(LOC, _Scorer(table), _ppmi({}, vocab), _Graph(),
                       _bins(vocab), lam=0.0, seed=0, arm="semantic")
    assert got[1] == pytest.approx(2 / 3)   # seed + one newly found gold


def test_oracle_anchors_exclude_the_systems_mistakes():
    # oracle and retrieved differ ONLY in whether a wrong proposal becomes an
    # anchor. d1 wins on cosine, so both propose it first; retrieved then
    # anchors on it and d1's heavy co-citation with d2/d3 drags them up ahead
    # of the remaining golds, while oracle refuses the bad anchor and keeps
    # conditioning on the seed. The gap IS the compounding-error measurement,
    # so it has to be strict.
    seed_paper = revelation_order(LOC["gold"], 0, "p1:0")[0]
    rest = [g for g in LOC["gold"] if g != seed_paper]
    # d1's cosine must beat gold-plus-seed-bonus, or neither arm ever takes
    # the bad anchor and the two are trivially equal.
    table = {"d1": 10.0, "d2": 0.10, "d3": 0.10}
    table.update(dict.fromkeys(LOC["gold"], 0.20))
    vocab = LOC["candidates"]
    ppmi = _ppmi({("d1", "d2"): 50.0, ("d1", "d3"): 50.0,
                  (seed_paper, rest[0]): 0.5, (seed_paper, rest[1]): 0.5}, vocab)
    args = (_Scorer(table), ppmi, _Graph(), _bins(vocab))
    orc = run_location(LOC, *args, lam=1.0, seed=0, arm="hybrid_oracle")
    ret = run_location(LOC, *args, lam=1.0, seed=0, arm="hybrid_retrieved")
    assert orc[3] > ret[3]


def test_recall_is_monotone_in_k():
    scorer = _Scorer({c: 1.0 / (i + 1) for i, c in enumerate(LOC["candidates"])})
    vocab = LOC["candidates"]
    got = run_location(LOC, scorer, _ppmi({}, vocab), _Graph(), _bins(vocab),
                       lam=0.0, seed=0, arm="semantic")
    vals = [got[k] for k in KS]
    assert vals == sorted(vals)
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_incremental_ppmi_sum_matches_a_direct_mean_over():
    # The running sum replaces PPMI.mean_over for speed; if it drifted, every
    # hybrid arm would be scoring something other than the documented formula.
    vocab = ["g1", "g2", "g3", "d1", "d2", "d3"]
    ppmi = _ppmi({("g1", "g2"): 2.0, ("g1", "g3"): 3.0, ("g2", "g3"): 5.0}, vocab)
    anchors = frozenset({"g1", "g2"})
    direct = {c: ppmi.mean_over(c, anchors) for c in vocab}
    total = np.zeros(len(vocab))
    for a in sorted(anchors):
        j = ppmi.index[a]
        lo, hi = ppmi.matrix.indptr[j], ppmi.matrix.indptr[j + 1]
        for col, val in zip(ppmi.matrix.indices[lo:hi].tolist(),
                            ppmi.matrix.data[lo:hi].tolist(), strict=True):
            total[col] += val
    got = {c: total[i] / len(anchors) for i, c in enumerate(vocab)}
    assert got == pytest.approx(direct)


def test_base_scores_gathers_the_pool_exactly_once():
    # This was a quadratic bug: the base vector was built as
    #   [scorer.pool_scores(ctx, cands)[c] for c in cands]
    # which re-ran the whole gather once per candidate -- 3.9 s per location
    # Counting the calls is the only cheap way
    # to keep it from creeping back.
    from locus.experiments.completion import base_scores

    calls = []

    class _Counting(_Scorer):
        def pool_scores(self, ctx, candidates):
            calls.append(ctx)
            return super().pool_scores(ctx, candidates)

    scorer = _Counting({c: 0.5 for c in LOC["candidates"]})
    got = base_scores(scorer, LOC)
    assert len(calls) == 1
    assert len(got) == len(LOC["candidates"])


def test_all_arms_share_one_base_vector():
    # The arms differ only in anchors, so recomputing the cosine per arm is
    # five times the work for identical numbers.
    from locus.experiments.completion import base_scores

    scorer = _Scorer({c: 0.5 for c in LOC["candidates"]})
    vocab = LOC["candidates"]
    shared = base_scores(scorer, LOC)
    args = (scorer, _ppmi({}, vocab), _Graph(), _bins(vocab))
    passed = run_location(LOC, *args, lam=0.0, seed=0, arm="semantic",
                          base=shared)
    computed = run_location(LOC, *args, lam=0.0, seed=0, arm="semantic")
    assert passed == computed
    # `Recalls` is a dict subclass, so `==` compares only the seed-inclusive
    # numbers; `.remaining` rides alongside and has to be checked by hand or
    # a passed-in base could silently change it.
    assert passed.remaining == computed.remaining


def test_random_arm_substitutes_the_seed_anchor_too():
    # If the random arm keeps the TRUE seed as its first anchor, it is not a
    # random-anchor control -- it is a control on anchors-after-the-first, and
    # since the seed carries most of the signal every arm scores the same R@1.
    scorer = _Scorer({"d1": 0.9, "d2": 0.8, "g1": 0.5, "g2": 0.1, "g3": 0.1})
    vocab = ["g1", "g2", "g3", "d1", "d2", "d3"]
    ppmi = _ppmi({("g1", "g2"): 9.0, ("g1", "g3"): 9.0, ("g2", "g3"): 9.0}, vocab)

    class _V:
        vocab = ("d3",)   # every random draw is a paper with no gold co-citation

    args = (scorer, ppmi, _V(), _bins(vocab))
    orc = run_location(LOC, *args, lam=1.0, seed=0, arm="hybrid_oracle")
    rnd = run_location(LOC, *args, lam=1.0, seed=0, arm="hybrid_random")
    assert orc[1] > rnd[1]


def test_selfseed_gets_no_free_recall_credit():
    # With no revealed seed the numerator is the predictions alone. Leaving
    # the seed in it would hand every arm 1/|G| for free and make the
    # self-seeded numbers silently comparable to the revealed-seed ones.
    scorer = _Scorer({"d1": 0.9, "d2": 0.8, "d3": 0.7,
                      "g1": 0.1, "g2": 0.05, "g3": 0.01})
    ppmi = _ppmi({}, ["g1", "g2", "g3", "d1", "d2", "d3"])
    got = run_location(LOC, scorer, ppmi, _Graph(), _bins(_Graph().vocab),
                       lam=1.0, seed=0, arm="semantic", reveal_seed=False)
    assert got[1] == pytest.approx(0.0)
    assert got[3] == pytest.approx(0.0)


def test_selfseed_first_pick_is_pure_base():
    # A_0 is empty, so step 1 must be the base ranking even for a hybrid arm.
    # If the seed were still added as an anchor, the huge PPMI between g1 and
    # its siblings would pull a gold paper to the top instead of d1.
    scorer = _Scorer({"d1": 0.9, "g1": 0.5, "g2": 0.1, "g3": 0.1,
                      "d2": 0.0, "d3": 0.0})
    ppmi = _ppmi({("g1", "g2"): 50.0, ("g1", "g3"): 50.0,
                  ("g2", "g3"): 50.0},
                 ["g1", "g2", "g3", "d1", "d2", "d3"])
    trace = []
    run_location(LOC, scorer, ppmi, _Graph(), _bins(_Graph().vocab),
                 lam=1.0, seed=0, arm="hybrid_retrieved", reveal_seed=False,
                 trace=trace)
    assert trace[0]["pick"] == "d1"
    assert trace[0]["n_anchors"] == 0


def test_selfseed_conditions_on_its_own_first_pick():
    # After picking g1 on base alone, PPMI(g1, g2) must lift g2 above the
    # distractors it was losing to. This is the whole self-seeded claim.
    scorer = _Scorer({"g1": 0.9, "d1": 0.5, "d2": 0.4, "d3": 0.3,
                      "g2": 0.1, "g3": 0.05})
    ppmi = _ppmi({("g1", "g2"): 50.0, ("g1", "g3"): 50.0},
                 ["g1", "g2", "g3", "d1", "d2", "d3"])
    trace = []
    got = run_location(LOC, scorer, ppmi, _Graph(), _bins(_Graph().vocab),
                       lam=1.0, seed=0, arm="hybrid_retrieved",
                       reveal_seed=False, trace=trace)
    assert trace[0]["pick"] == "g1"
    assert trace[1]["pick"] in ("g2", "g3")
    assert got[3] == pytest.approx(1.0)


def test_trace_records_seed_quality_and_anchor_counts():
    # The seed-quality curve is built off these fields, so they have to
    # distinguish "how many anchors" from "how many were right".
    scorer = _Scorer({"d1": 0.9, "g1": 0.5, "g2": 0.4, "g3": 0.3,
                      "d2": 0.0, "d3": 0.0})
    ppmi = _ppmi({}, ["g1", "g2", "g3", "d1", "d2", "d3"])
    trace = []
    run_location(LOC, scorer, ppmi, _Graph(), _bins(_Graph().vocab),
                 lam=1.0, seed=0, arm="hybrid_retrieved", reveal_seed=False,
                 trace=trace)
    # step 1 picks d1 (wrong), step 2 picks g1 (right).
    assert trace[0]["pick"] == "d1" and trace[0]["correct"] is False
    assert trace[1]["correct"] is True
    # entering step 2 the arm holds one anchor, none of them correct.
    assert trace[1]["n_anchors"] == 1
    assert trace[1]["n_correct"] == 0
    # entering step 3 it holds two, one of which is correct.
    assert trace[2]["n_anchors"] == 2
    assert trace[2]["n_correct"] == 1


def test_remaining_recall_excludes_the_seed_from_both_sides():
    # The seed-inclusive metric hands every arm 1/|G| for free. The whole
    # point of `.remaining` is that the free credit is gone from BOTH the
    # numerator and the denominator, so this pins the two against each other
    # on a location where the arm finds exactly one of the two remaining
    # golds: 2/3 inclusive against 1/2 remaining. A denominator left at |G|
    # would read 1/3 here, and a numerator that still counted the seed 1.0.
    seed_paper = revelation_order(LOC["gold"], 0, "p1:0")[0]
    other_gold = next(g for g in LOC["gold"] if g != seed_paper)
    table = dict.fromkeys(LOC["candidates"], 0.1)
    table[seed_paper] = 0.99
    table[other_gold] = 0.98
    vocab = LOC["candidates"]
    got = run_location(LOC, _Scorer(table), _ppmi({}, vocab), _Graph(),
                       _bins(vocab), lam=0.0, seed=0, arm="semantic")
    assert got[1] == pytest.approx(2 / 3)
    assert got.remaining[1] == pytest.approx(1 / 2)


def test_remaining_recall_is_undefined_without_a_revealed_seed():
    # Self-seeded, "the remaining members" names nothing: there is no seed to
    # remove. Returning the inclusive numbers under a second name would make
    # the two modes silently comparable, which is the trap the recall credit
    # already avoids -- so this has to be None, not a number.
    scorer = _Scorer({"d1": 0.9, "g1": 0.5, "g2": 0.1, "g3": 0.1,
                      "d2": 0.0, "d3": 0.0})
    vocab = LOC["candidates"]
    got = run_location(LOC, scorer, _ppmi({}, vocab), _Graph(), _bins(vocab),
                       lam=1.0, seed=0, arm="semantic", reveal_seed=False)
    assert got.remaining is None


def test_remaining_and_inclusive_are_two_views_of_one_run():
    # Both are scored off the SAME picks -- the arm is not re-run -- so
    # inclusive must be recoverable from remaining exactly. If `.remaining`
    # were ever computed after the seed joined `found`, or from a separate
    # pass, this identity is what breaks.
    scorer = _Scorer({"d1": 0.9, "d2": 0.8, "g1": 0.5, "g2": 0.1, "g3": 0.1})
    vocab = ["g1", "g2", "g3", "d1", "d2", "d3"]
    ppmi = _ppmi({("g1", "g2"): 9.0, ("g1", "g3"): 9.0, ("g2", "g3"): 9.0}, vocab)
    n = len(LOC["gold"])
    for arm in ("semantic", "hybrid_retrieved", "hybrid_oracle"):
        got = run_location(LOC, scorer, ppmi, _Graph(), _bins(vocab),
                           lam=1.0, seed=0, arm=arm)
        for k in KS:
            found = got.remaining[k] * (n - 1)
            assert got[k] == pytest.approx((1 + found) / n), (arm, k)


def test_remaining_recall_is_monotone_and_bounded():
    scorer = _Scorer({c: 1.0 / (i + 1) for i, c in enumerate(LOC["candidates"])})
    vocab = LOC["candidates"]
    got = run_location(LOC, scorer, _ppmi({}, vocab), _Graph(), _bins(vocab),
                       lam=0.0, seed=0, arm="semantic")
    vals = [got.remaining[k] for k in KS]
    assert vals == sorted(vals)
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_by_set_size_buckets_by_citation_count_and_caps_the_tail():
    # Sizes 2, 3 and 4 get their own buckets; everything from 5 up lands in
    # one, because 369 of 22,274 test locations carry ten or more citations
    # and a bucket per size would be reporting noise.
    from locus.experiments.completion import by_set_size

    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    groups = ["a", "b", "c", "d", "e"]
    sizes = [2, 3, 4, 5, 40]
    got = by_set_size(values, groups, sizes)
    assert sorted(got) == [2, 3, 4, 5]
    assert got[5]["rows"] == 2            # sizes 5 and 40 share a bucket
    assert got[5]["mean"] == pytest.approx(4.5)


def test_by_set_size_macro_averages_within_a_bucket():
    # paper_mean inside each bucket, matching every other number in the
    # report. One citing paper with three locations must not outvote three
    # papers with one each -- a plain mean over rows would give 0.55, the
    # macro-average gives 0.625.
    from locus.experiments.completion import by_set_size

    values = [1.0, 1.0, 1.0, 0.0, 0.5, 1.0]
    groups = ["heavy", "heavy", "heavy", "p1", "p2", "p3"]
    got = by_set_size(values, groups, [2] * 6)
    assert got[2]["rows"] == 6
    assert got[2]["mean"] == pytest.approx(0.625)


def test_base_is_z_scored_within_the_candidate_list_by_default():
    # Eq. 1 adds lambda * PPMI to a STANDARDISED base. On raw cosine a PPMI
    # term of 1.0 at lambda = 1 outweighs a 0.8 cosine gap and the co-cited
    # gold is picked; on z-scores (sd ~0.35 here) the gap is ~2.3 units and
    # the distractor keeps the top spot. The two settings must disagree on
    # this pick, and the paper's setting must be the default.
    vocab = ["g1", "g2", "g3", "d1", "d2", "d3"]
    seed_paper = revelation_order(LOC["gold"], 0, "p1:0")[0]
    sibling = next(g for g in LOC["gold"] if g != seed_paper)
    scorer = _Scorer({"d1": 0.9, "d2": 0.8, "g1": 0.1, "g2": 0.1, "g3": 0.1,
                      "d3": 0.1})
    ppmi = _ppmi({(seed_paper, sibling): 1.0}, vocab)
    args = (scorer, ppmi, _Graph(), _bins(vocab))

    raw_trace: list = []
    run_location(LOC, *args, lam=1.0, seed=0, arm="hybrid_retrieved",
                 trace=raw_trace, standardize="none")
    z_trace: list = []
    run_location(LOC, *args, lam=1.0, seed=0, arm="hybrid_retrieved",
                 trace=z_trace)
    assert raw_trace[0]["pick"] == sibling
    assert z_trace[0]["pick"] == "d1"


def test_z_scoring_never_changes_the_semantic_arm():
    # z-scoring is monotone, so an arm that adds no PPMI must produce the
    # identical sequence of picks and the identical recall under both.
    vocab = ["g1", "g2", "g3", "d1", "d2", "d3"]
    scorer = _Scorer({"d1": 0.9, "g2": 0.85, "d2": 0.8, "g1": 0.5, "g3": 0.2,
                      "d3": 0.1})
    ppmi = _ppmi({}, vocab)
    args = (scorer, ppmi, _Graph(), _bins(vocab))
    raw_trace: list = []
    raw = run_location(LOC, *args, lam=1.0, seed=0, arm="semantic",
                       trace=raw_trace, standardize="none")
    z_trace: list = []
    z = run_location(LOC, *args, lam=1.0, seed=0, arm="semantic",
                     trace=z_trace, standardize="zscore")
    assert [t["pick"] for t in raw_trace] == [t["pick"] for t in z_trace]
    assert dict(raw) == dict(z)


def test_a_constant_base_z_scores_to_zeros_not_nan():
    # A location whose candidates all tie on cosine has zero variance. Dividing
    # by it would put NaN into every score and argmax would pick index 0
    # forever; the pool z-score convention (core.rescore) returns zeros.
    vocab = ["g1", "g2", "g3", "d1", "d2", "d3"]
    scorer = _Scorer(dict.fromkeys(vocab, 0.5))
    seed_paper = revelation_order(LOC["gold"], 0, "p1:0")[0]
    sibling = next(g for g in LOC["gold"] if g != seed_paper)
    ppmi = _ppmi({(seed_paper, sibling): 1.0}, vocab)
    trace: list = []
    got = run_location(LOC, scorer, ppmi, _Graph(), _bins(vocab), lam=1.0,
                       seed=0, arm="hybrid_retrieved", trace=trace)
    assert trace[0]["pick"] == sibling
    assert all(v == v for v in got.values())


def test_unknown_standardize_is_refused():
    vocab = ["g1", "g2", "g3", "d1", "d2", "d3"]
    scorer = _Scorer(dict.fromkeys(vocab, 0.5))
    with pytest.raises(ValueError, match="standardize"):
        run_location(LOC, scorer, _ppmi({}, vocab), _Graph(), _bins(vocab),
                     lam=1.0, seed=0, arm="semantic", standardize="minmax")


def test_experiment_b_intervals_use_the_papers_thousand_resamples():
    # Section 4.2 says every interval is a 1000-resample paper-level bootstrap.
    # Sequential completion ran at 500 until 2026-09-05; the constant is pinned so it
    # cannot quietly diverge from the other five entry points again.
    from locus.experiments import completion
    assert completion.N_BOOT == 1000
