from dataclasses import replace

from locus.core.pools import build_rank_items, build_swap_pairs
from locus.core.types import ContextRec


def _rec(cid, refid, loc):
    return ContextRec(
        context_id=f"{cid}_{refid}_0",
        citing_id=cid,
        refid=refid,
        raw="",
        masked="",
        location=loc,
    )


def _paper_with(n_locations, per_location):
    recs, r = [], 0
    for loc in range(n_locations):
        for _ in range(per_location):
            recs.append(_rec("P", str(r), loc))
            r += 1
    return {"P": recs}


class TestRankItems:
    def test_pool_has_exact_size_and_contains_gold(self):
        items, _ = build_rank_items(_paper_with(6, 2), pool_size=5, seed=0)
        assert items
        for it in items:
            assert len(it.candidates) == 5
            assert it.gold in it.candidates
            assert len(set(it.candidates)) == 5

    def test_distractors_never_come_from_the_query_location(self):
        index = _paper_with(6, 2)
        golds_by_loc = {}
        for r in index["P"]:
            golds_by_loc.setdefault(r.location, set()).add(r.refid)
        by_ctx = {r.context_id: r for r in index["P"]}
        items, _ = build_rank_items(index, pool_size=5, seed=0)
        for it in items:
            own_loc = by_ctx[it.context_id].location
            distractors = set(it.candidates) - {it.gold}
            assert not (distractors & golds_by_loc[own_loc])

    def test_papers_with_too_few_distractors_are_dropped(self):
        # 2 locations x 1 ref = 1 available distractor, need 4
        items, drops = build_rank_items(_paper_with(2, 1), pool_size=5, seed=0)
        assert items == []
        assert drops["too_few_distractors"] == 2

    def test_single_location_paper_is_reported_as_such(self):
        items, drops = build_rank_items(_paper_with(1, 6), pool_size=5, seed=0)
        assert items == []
        assert drops["single_location"] == 6
        assert drops["too_few_distractors"] == 0

    def test_alignment_failures_are_dropped_and_counted(self):
        index = _paper_with(6, 2)
        index["P"][0] = replace(index["P"][0], alignment_ok=False)
        items, drops = build_rank_items(index, pool_size=5, seed=0)
        assert drops["alignment_failed"] == 1
        assert all(it.context_id != index["P"][0].context_id for it in items)

    def test_drop_reasons_account_for_every_context(self):
        index = _paper_with(6, 2)
        items, drops = build_rank_items(index, pool_size=5, seed=0)
        assert sum(drops.values()) == len(index["P"])
        assert drops["kept"] == len(items)

    def test_deterministic_under_seed(self):
        a, _ = build_rank_items(_paper_with(6, 2), pool_size=5, seed=7)
        b, _ = build_rank_items(_paper_with(6, 2), pool_size=5, seed=7)
        assert a == b

    def test_seed_changes_the_sample(self):
        a, _ = build_rank_items(_paper_with(9, 2), pool_size=5, seed=1)
        b, _ = build_rank_items(_paper_with(9, 2), pool_size=5, seed=2)
        assert a != b


class TestSwapPairs:
    def test_pairs_have_disjoint_gold_sets(self):
        index = _paper_with(4, 2)
        by_ctx = {r.context_id: r for r in index["P"]}
        golds_by_loc = {}
        for r in index["P"]:
            golds_by_loc.setdefault(r.location, set()).add(r.refid)
        for p in build_swap_pairs(index, k=5, seed=0):
            la = by_ctx[p.ctx_a].location
            lb = by_ctx[p.ctx_b].location
            assert la != lb
            assert not (golds_by_loc[la] & golds_by_loc[lb])

    def test_capped_at_k_per_paper(self):
        pairs = build_swap_pairs(_paper_with(20, 1), k=5, seed=0)
        assert len(pairs) == 5

    def test_single_location_paper_yields_nothing(self):
        assert build_swap_pairs(_paper_with(1, 3), k=5, seed=0) == []

    def test_deterministic_under_seed(self):
        a = build_swap_pairs(_paper_with(10, 1), k=5, seed=3)
        b = build_swap_pairs(_paper_with(10, 1), k=5, seed=3)
        assert a == b
