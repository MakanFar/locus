# tests/data/test_corpus.py
import json
import pathlib
from dataclasses import replace

import pytest

from locus import config
from locus.core.types import (
    ContextRec,
    anchors,
    anchors_union,
    clustering_disagreement,
    gold_sets,
)
from locus.data import corpus
from locus.data.corpus import build_index, load_papers


def _rec(cid, refid, loc):
    return ContextRec(
        context_id=f"{cid}_{refid}_0",
        citing_id=cid,
        refid=refid,
        raw="",
        masked="",
        location=loc,
    )


def _ctx(cid, refid, raw, masked):
    return {"citing_id": cid, "refid": refid, "raw": raw, "masked_text": masked}


_W1 = "we adopt the gaia dr2 parallax zero point offset reported in that work here"
_W2 = "an entirely unrelated sentence about lattice quantum chromodynamics follows"


def _write_two_paper_corpus(tmp_path):
    contexts = {
        "P1_a": _ctx("P1", "R1", f"{_W1} [4]", f"{_W1} TARGETCIT"),
        "P1_b": _ctx("P1", "R2", f"{_W1} [5]", f"{_W1} TARGETCIT"),
        "P2_a": _ctx("P2", "R3", f"{_W2} [7]", f"{_W2} TARGETCIT"),
    }
    cpath = tmp_path / "contexts.json"
    cpath.write_text(json.dumps(contexts), encoding="utf-8")
    spath = tmp_path / "split.json"
    spath.write_text(
        json.dumps([{"context_id": k, "positive_ids": []} for k in contexts]),
        encoding="utf-8",
    )
    return cpath, spath


def _write_reopened_corpus(tmp_path):
    # P1's run closes when P2 opens, then P1 appears again.
    contexts = {
        "P1_a": _ctx("P1", "R1", f"{_W1} [4]", f"{_W1} TARGETCIT"),
        "P2_a": _ctx("P2", "R3", f"{_W2} [7]", f"{_W2} TARGETCIT"),
        "P1_b": _ctx("P1", "R2", f"{_W1} [5]", f"{_W1} TARGETCIT"),
    }
    cpath = tmp_path / "contexts.json"
    cpath.write_text(json.dumps(contexts), encoding="utf-8")
    spath = tmp_path / "split.json"
    spath.write_text(
        json.dumps([{"context_id": k, "positive_ids": []} for k in contexts]),
        encoding="utf-8",
    )
    return cpath, spath


class TestGoldSets:
    def test_groups_refids_by_location(self):
        recs = [_rec("P", "1", 0), _rec("P", "2", 0), _rec("P", "3", 1)]
        assert gold_sets(recs) == {0: {"1", "2"}, 1: {"3"}}

    def test_empty(self):
        assert gold_sets([]) == {}


class TestAnchors:
    def test_excludes_own_target(self):
        recs = [_rec("P", "1", 0), _rec("P", "2", 0), _rec("P", "3", 0)]
        golds = gold_sets(recs)
        assert anchors(recs[0], golds) == {"2", "3"}

    def test_singleton_location_has_no_anchors(self):
        recs = [_rec("P", "1", 0)]
        assert anchors(recs[0], gold_sets(recs)) == set()


class TestAnchorsUnion:
    def test_equals_clustering_when_marker_anchors_empty(self):
        recs = [_rec("P", "1", 0), _rec("P", "2", 0), _rec("P", "3", 0)]
        golds = gold_sets(recs)
        assert anchors_union(recs[0], golds) == anchors(recs[0], golds) == {"2", "3"}

    def test_picks_up_a_marker_anchor_clustering_missed(self):
        # "9" is not in this context's location cluster at all (it belongs to
        # no location's gold set here), but an OTHERCIT marker in the window
        # resolved to it -- the marker route recovers it, clustering can't.
        recs = [_rec("P", "1", 0), _rec("P", "2", 0)]
        recs[0] = replace(recs[0], marker_anchors=frozenset({"9"}))
        golds = gold_sets(recs)
        assert anchors(recs[0], golds) == {"2"}
        assert anchors_union(recs[0], golds) == {"2", "9"}

    def test_excludes_own_target_even_with_marker_anchors(self):
        recs = [_rec("P", "1", 0), _rec("P", "2", 0), _rec("P", "3", 1)]
        recs[0] = replace(recs[0], marker_anchors=frozenset({"2"}))
        golds = gold_sets(recs)
        result = anchors_union(recs[0], golds)
        assert "1" not in result
        assert result == {"2"}


class TestClusteringDisagreement:
    def test_zero_when_markers_agree_with_the_cluster(self):
        recs = [_rec("P", "1", 0), _rec("P", "2", 0)]
        recs[0] = replace(recs[0], marker_anchors=frozenset({"2"}))
        assert clustering_disagreement(recs[0], gold_sets(recs)) == 0

    def test_counts_markers_the_cluster_omitted(self):
        # marker says "3" is in this window, clustering put it elsewhere
        recs = [_rec("P", "1", 0), _rec("P", "2", 0), _rec("P", "3", 1)]
        recs[0] = replace(recs[0], marker_anchors=frozenset({"2", "3"}))
        assert clustering_disagreement(recs[0], gold_sets(recs)) == 1

    def test_zero_when_no_markers_recovered(self):
        recs = [_rec("P", "1", 0), _rec("P", "2", 0)]
        assert clustering_disagreement(recs[0], gold_sets(recs)) == 0


class TestBuildIndexCompleteness:
    def test_raises_when_stream_yields_fewer_contexts_than_the_split_wants(
        self, monkeypatch
    ):
        # stream_dict returns silently at EOF and on an unterminated record, so
        # a truncated contexts.json would otherwise yield a smaller corpus with
        # every rate-based gate still passing. No real dataset is touched here:
        # both _split_context_ids and stream_dict are monkeypatched.
        import locus.data.corpus as corpus_mod

        monkeypatch.setattr(
            corpus_mod, "_split_context_ids", lambda split: {"a", "b", "c"}
        )
        monkeypatch.setattr(
            corpus_mod,
            "stream_dict",
            lambda path: iter(
                [("a", {"citing_id": "P", "refid": "1", "raw": "", "masked_text": ""})]
            ),
        )
        with pytest.raises(SystemExit):
            build_index("test")


class TestLoadPapers:
    FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "papers_sample.json"

    def _ids(self, n=None):
        with open(self.FIXTURE, encoding="utf-8") as f:
            keys = list(json.load(f))
        return set(keys if n is None else keys[:n])

    def test_returns_every_requested_paper(self, monkeypatch):
        monkeypatch.setattr(config, "papers", lambda: self.FIXTURE)
        ids = self._ids(3)
        out = load_papers(ids)
        assert set(out) == ids

    def test_a_missing_paper_is_fatal(self, monkeypatch):
        # build_index guards the identical failure mode for contexts.json
        # (found != len(want) -> SystemExit). Without the same guard here a
        # truncated papers.json silently shrinks hard_ids and headline_ids
        # while the 40-85% hard-fraction gate still passes.
        monkeypatch.setattr(config, "papers", lambda: self.FIXTURE)
        with pytest.raises(SystemExit, match=r"papers\.json"):
            load_papers(self._ids(2) | {"not-a-real-paper-id"})


class TestIterPapers:
    def test_agrees_with_build_index_on_a_fixture_split(self, monkeypatch, tmp_path):
        # Same inputs must give the same index whether accumulated or streamed.
        contexts, split = _write_two_paper_corpus(tmp_path)
        monkeypatch.setattr(config, "contexts", lambda: contexts)
        monkeypatch.setattr(config, "split_file", lambda s: split)
        assert dict(corpus.iter_papers("test")) == corpus.build_index("test")

    def test_yields_one_paper_at_a_time_in_file_order(self, monkeypatch, tmp_path):
        contexts, split = _write_two_paper_corpus(tmp_path)
        monkeypatch.setattr(config, "contexts", lambda: contexts)
        monkeypatch.setattr(config, "split_file", lambda s: split)
        assert [cid for cid, _ in corpus.iter_papers("test")] == ["P1", "P2"]

    def test_a_reopened_citing_id_is_fatal(self, monkeypatch, tmp_path):
        # A split run would build that paper's bibliography from only part of
        # its contexts, producing plausible-looking wrong anchors instead of
        # an error. Contiguity is a property of today's file, so it is checked.
        contexts, split = _write_reopened_corpus(tmp_path)
        monkeypatch.setattr(config, "contexts", lambda: contexts)
        monkeypatch.setattr(config, "split_file", lambda s: split)
        with pytest.raises(SystemExit, match="contiguous"):
            list(corpus.iter_papers("test"))

    def test_keep_text_false_blanks_text_but_keeps_structure(self, monkeypatch, tmp_path):
        contexts, split = _write_two_paper_corpus(tmp_path)
        monkeypatch.setattr(config, "contexts", lambda: contexts)
        monkeypatch.setattr(config, "split_file", lambda s: split)
        full = {c: r for c, r in corpus.iter_papers("test")}
        lean = {c: r for c, r in corpus.iter_papers("test", keep_text=False)}
        for cid, recs in lean.items():
            assert [r.raw for r in recs] == [""] * len(recs)
            assert [r.masked for r in recs] == [""] * len(recs)
            # locations still come from the real text, so they must match
            assert [r.location for r in recs] == [r.location for r in full[cid]]
            assert [r.refid for r in recs] == [r.refid for r in full[cid]]
            assert [r.alignment_ok for r in recs] == [r.alignment_ok for r in full[cid]]
            assert [r.marker_anchors for r in recs] == [r.marker_anchors for r in full[cid]]

    def test_a_partially_consumed_iterator_does_not_assert_completeness(
        self, monkeypatch, tmp_path
    ):
        # The found-vs-wanted check belongs at the end of a full pass. Taking
        # one paper and walking away is not evidence the file is truncated.
        contexts, split = _write_two_paper_corpus(tmp_path)
        monkeypatch.setattr(config, "contexts", lambda: contexts)
        monkeypatch.setattr(config, "split_file", lambda s: split)
        it = corpus.iter_papers("test")
        assert next(it)[0] == "P1"
        it.close()
