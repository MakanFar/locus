import hashlib
import json
import pickle
from dataclasses import replace

import pytest

from locus import config
from locus.core.pools import RankItem, SwapPair, build_rank_items
from locus.core.types import ContextRec
from locus.data.export import (
    anchor_rows,
    export,
    load_anchors,
    load_rank,
    load_swap,
    load_target_counts,
    rank_rows,
    write_jsonl,
)


def _rec(cid, refid, loc, markers=(), ok=True):
    return ContextRec(
        context_id=f"{cid}_{refid}_0", citing_id=cid, refid=refid,
        raw="some window text", masked="some window text",
        location=loc, alignment_ok=ok, marker_anchors=frozenset(markers),
    )


def _index(n_locations=6, per_location=2):
    recs, r = [], 0
    for loc in range(n_locations):
        for _ in range(per_location):
            recs.append(_rec("P", str(r), loc))
            r += 1
    return {"P": recs}


def _write_pickles(work, index, pool_size=5):
    items, _ = build_rank_items(index, pool_size=pool_size, seed=0)
    pairs = [SwapPair("P", "P_0_0", "P_2_0", "0", "2")]
    hard = sorted({it.context_id for it in items[:3]})
    headline = hard[:2]
    for name, obj in (("rank_items", items), ("swap_pairs", pairs),
                      ("hard_ids", hard), ("headline_ids", headline),
                      ("index", index)):
        (work / f"{name}_test.pkl").write_bytes(pickle.dumps(obj, protocol=5))
    return items, pairs, set(hard), set(headline)


class TestRowShape:
    def test_candidate_order_is_preserved_not_sorted(self):
        # Sorting candidates changes no metric -- scoring is keyed by
        # candidate -- so nothing downstream would fail. It would silently
        # publish a different artefact from the one the paper measured.
        item = RankItem("c1", "P", "g", ("z", "a", "g", "m"))
        (row,) = rank_rows([item], set(), set())
        assert row["candidates"] == ["z", "a", "g", "m"]

    def test_slice_membership_becomes_flags(self):
        item = RankItem("c1", "P", "g", ("g", "d"))
        (row,) = rank_rows([item], hard={"c1"}, headline=set())
        assert (row["hard"], row["headline"]) == (True, False)

    def test_rows_carry_no_text(self):
        # The whole point of the export: ids are ours to publish, the upstream
        # corpus's abstracts and context windows are not. A row that leaked
        # `raw` or `masked` would make the artefact unredistributable.
        index = _index()
        for row in anchor_rows(index):
            for value in row.values():
                assert "some window text" not in json.dumps(value)

    def test_anchor_routes_are_reported_alongside_their_union(self):
        # Neither route subsumes the other, so a consumer must be able to see
        # the divergence rather than inherit our choice silently.
        index = {"P": [_rec("P", "1", 0, markers=["9"]), _rec("P", "2", 0)]}
        rows = {r["context_id"]: r for r in anchor_rows(index)}
        r = rows["P_1_0"]
        assert r["anchors_clustering"] == ["2"]
        assert r["anchors_marker"] == ["9"]
        assert r["anchors"] == ["2", "9"]

    def test_anchors_row_exists_for_every_context_including_failures(self):
        # build_rank_items drops alignment failures, so they have no rank row.
        # They still carry anchors, and they still count toward the popularity
        # null's target frequencies.
        index = {"P": [_rec("P", "1", 0, ok=False), _rec("P", "2", 0)]}
        rows = list(anchor_rows(index))
        assert {r["context_id"] for r in rows} == {"P_1_0", "P_2_0"}
        assert [r["alignment_ok"] for r in rows] == [False, True]


class TestWriteAndLoad:
    def test_sha256_is_of_the_bytes_actually_written(self, tmp_path):
        p = tmp_path / "a.jsonl"
        n, sha = write_jsonl(p, [{"a": 1}, {"a": 2}])
        assert n == 2
        assert sha == hashlib.sha256(p.read_bytes()).hexdigest()

    def test_rank_round_trips_through_jsonl(self, tmp_path):
        items = [RankItem("c1", "P", "g", ("g", "d1")),
                 RankItem("c2", "P", "h", ("d2", "h"))]
        p = tmp_path / "r.jsonl"
        write_jsonl(p, rank_rows(items, set(), set()))
        assert load_rank(p) == items

    def test_swap_round_trips_through_jsonl(self, tmp_path):
        pairs = [SwapPair("P", "a", "b", "ga", "gb")]
        p = tmp_path / "s.jsonl"
        from locus.data.export import swap_rows
        write_jsonl(p, swap_rows(pairs))
        assert load_swap(p) == pairs

    def test_target_counts_come_from_every_context_not_every_pool(self, tmp_path):
        # build_day1 counts targets over the index, not over rank-item golds,
        # because swap pairs are built from records rank-item construction
        # drops. Counting only aligned contexts here would leave popularity at
        # log1p(0) for those pairs -- byte-identical to the constant scorer,
        # collapsing two null controls into one.
        index = {"P": [_rec("P", "1", 0, ok=False), _rec("P", "1", 1),
                       _rec("P", "2", 1)]}
        p = tmp_path / "a.jsonl"
        write_jsonl(p, anchor_rows(index))
        assert load_target_counts(p) == {"1": 2, "2": 1}

    def test_load_anchors_is_a_drop_in_for_anchors_union(self, tmp_path):
        from locus.core.types import anchors_union, gold_sets
        index = _index()
        p = tmp_path / "a.jsonl"
        write_jsonl(p, anchor_rows(index))
        got = load_anchors(p)
        golds = gold_sets(index["P"])
        for rec in index["P"]:
            assert got[rec.context_id] == frozenset(anchors_union(rec, golds))


class TestExport:
    def test_manifest_floor_follows_the_data_not_our_config(self, tmp_path):
        # The floor moves with pool size (0.29290 at k=10, 0.31433 at k=9), and
        # the consumer checks it by equality. Reading k from config.POOL_SIZE
        # would publish a floor describing OUR settings rather than the file
        # they hold -- and the two differ exactly when it matters.
        work, out = tmp_path / "work", tmp_path / "out"
        work.mkdir()
        _write_pickles(work, _index(), pool_size=5)
        m = export("test", work, out)
        assert m["floors"]["pool_size"] == 5
        assert m["floors"]["rank_constant_mrr"] == repr(
            sum(1.0 / i for i in range(1, 6)) / 5
        )
        assert config.POOL_SIZE != 5, "fixture no longer exercises the gap"

    def test_mixed_pool_sizes_are_refused(self, tmp_path):
        # A set with two pool sizes has two floors, so it has none to publish.
        work, out = tmp_path / "work", tmp_path / "out"
        work.mkdir()
        items, _, _, _ = _write_pickles(work, _index())
        items[0] = replace(items[0], candidates=items[0].candidates[:-1])
        (work / "rank_items_test.pkl").write_bytes(pickle.dumps(items, protocol=5))
        with pytest.raises(ValueError, match="mixed sizes"):
            export("test", work, out)

    def test_manifest_declares_that_no_text_is_included(self, tmp_path):
        work, out = tmp_path / "work", tmp_path / "out"
        work.mkdir()
        _write_pickles(work, _index())
        m = export("test", work, out)
        assert "IDENTIFIERS only" in m["source"]["note"]
