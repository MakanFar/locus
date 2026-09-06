import pickle

import polars as pl
import pytest

from locus import config
from locus.build import embed_inputs
from locus.build.embed_inputs import (
    build_frame,
    collect_keys,
    compose_paper_text,
)
from locus.core.pools import RankItem, SwapPair
from locus.core.types import ContextRec


def _rec(cid, refid, loc, masked="a window"):
    return ContextRec(
        context_id=f"{cid}_{refid}_{loc}", citing_id=cid, refid=refid,
        raw="raw text", masked=masked, location=loc,
    )


def _fixture(tmp_path, monkeypatch, *, papers=None, index=None, masked="a window"):
    """Frozen pools on disk plus monkeypatched loaders; returns nothing."""
    items = [RankItem("c_g1_0", "c", "g1", ("g1", "d1", "d2"))]
    pairs = [SwapPair("c", "c_g1_0", "c_g2_1", "g1", "g2")]
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    with open(tmp_path / "rank_items_test.pkl", "wb") as f:
        pickle.dump(items, f)
    with open(tmp_path / "swap_pairs_test.pkl", "wb") as f:
        pickle.dump(pairs, f)
    if index is None:
        index = {"c": [_rec("c", "g1", 0, masked), _rec("c", "g2", 1, masked)]}
    monkeypatch.setattr(embed_inputs, "load_index", lambda path: index)
    if papers is None:
        papers = {
            k: {"title": f"T{k}", "abstract": f"A{k}", "authors": []}
            for k in ("g1", "g2", "d1", "d2")
        }
    monkeypatch.setattr(embed_inputs, "load_papers", lambda ids, **_: papers)


def test_compose_paper_text_is_the_production_recipe():
    # Literal ' [SEP] ', not the tokenizer's special token: this is the string
    # the verified production run embedded, and the vectors are only
    # comparable to it if the byte sequence matches.
    assert compose_paper_text("Title", "Abstract") == "Title [SEP] Abstract"
    assert compose_paper_text("", "Abstract") == " [SEP] Abstract"


def test_collect_keys_takes_swap_golds_and_contexts_too():
    # A swap pair's golds and contexts are not necessarily in any rank pool --
    # swap pairs are built from records build_rank_items drops.
    items = [RankItem("c_g1_0", "c", "g1", ("g1", "d1"))]
    pairs = [SwapPair("z", "z_x_0", "z_y_1", "x", "y")]
    papers, contexts = collect_keys(items, pairs)
    assert papers == {"g1", "d1", "x", "y"}
    assert contexts == {"c_g1_0", "z_x_0", "z_y_1"}


def test_frame_covers_every_candidate_and_context(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    frame, manifest = build_frame("test")
    papers = set(frame.filter(frame["kind"] == "paper")["key"])
    contexts = set(frame.filter(frame["kind"] == "context")["key"])
    assert papers == {"g1", "g2", "d1", "d2"}
    assert contexts == {"c_g1_0", "c_g2_1"}
    assert manifest["rows"] == frame.height == 6


def test_abstract_less_candidate_is_still_embedded(tmp_path, monkeypatch):
    # The production recipe gates on abstracts of >50 chars. Applying that gate
    # here would drop d2 from the pool, shrinking it from 3 to 2 and moving the
    # chance floor H_k/k out from under the frozen build gate.
    papers = {
        "g1": {"title": "T", "abstract": "A", "authors": []},
        "g2": {"title": "T", "abstract": "A", "authors": []},
        "d1": {"title": "T", "abstract": "A", "authors": []},
        "d2": {"title": "Only a title", "abstract": "", "authors": []},
    }
    _fixture(tmp_path, monkeypatch, papers=papers)
    frame, manifest = build_frame("test")
    row = frame.filter((frame["kind"] == "paper") & (frame["key"] == "d2"))
    assert row.height == 1
    assert row["text"][0] == "Only a title [SEP] "
    assert manifest["papers_without_abstract"] == 1


def test_context_missing_from_index_is_refused(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch, index={"c": [_rec("c", "g1", 0)]})
    with pytest.raises(SystemExit, match=r"absent from index_test\.pkl"):
        build_frame("test")


def test_empty_masked_window_is_refused(tmp_path, monkeypatch):
    # Every empty query embeds to the same vector, which ties its whole pool.
    _fixture(tmp_path, monkeypatch, masked="   ")
    with pytest.raises(SystemExit, match="empty"):
        build_frame("test")


def _sibling_fixture(tmp_path, monkeypatch):
    # p1 has two locations: loc 0 targets g1 and d1, loc 1 targets g2. So the
    # anchor set of context "c_g1_0" is {d1} by the clustering route.
    recs = [
        ContextRec("c_g1_0", "c", "g1", "raw", "window one", 0),
        ContextRec("c_d1_0", "c", "d1", "raw", "window two", 0),
        ContextRec("c_g2_1", "c", "g2", "raw", "window three", 1),
    ]
    papers = {k: {"title": f"Title {k}", "abstract": f"Abs {k}", "authors": []}
              for k in ("g1", "g2", "d1", "d2")}
    _fixture(tmp_path, monkeypatch, papers=papers, index={"c": recs})
    return papers


def test_siblings_variant_appends_anchor_titles(tmp_path, monkeypatch):
    _sibling_fixture(tmp_path, monkeypatch)
    frame, manifest = build_frame("test", variant="siblings")
    texts = dict(zip(frame["key"], frame["text"], strict=True))
    # c_g1_0's anchor is d1, so d1's title joins the query; its own gold does not.
    assert texts["c_g1_0"] == "window one Title d1"
    assert "Title g1" not in texts["c_g1_0"]
    assert manifest["variant"] == "siblings"
    assert manifest["contexts_with_sibling_titles"] >= 1


def test_plain_variant_leaves_the_query_untouched(tmp_path, monkeypatch):
    _sibling_fixture(tmp_path, monkeypatch)
    frame, _ = build_frame("test", variant="plain")
    texts = dict(zip(frame["key"], frame["text"], strict=True))
    assert texts["c_g1_0"] == "window one"


def test_anchor_titles_are_emitted_in_sorted_order(tmp_path, monkeypatch):
    # An anchor set is a frozenset; iterating one directly would make the query
    # text depend on PYTHONHASHSEED and the run irreproducible.
    recs = [
        ContextRec("c_g1_0", "c", "g1", "raw", "w", 0),
        ContextRec("c_zz_0", "c", "zz", "raw", "w", 0),
        ContextRec("c_aa_0", "c", "aa", "raw", "w", 0),
        ContextRec("c_g2_1", "c", "g2", "raw", "w", 1),
    ]
    papers = {k: {"title": f"T{k}", "abstract": "a", "authors": []}
              for k in ("g1", "g2", "zz", "aa", "d1", "d2")}
    _fixture(tmp_path, monkeypatch, papers=papers, index={"c": recs})
    frame, _ = build_frame("test", variant="siblings")
    texts = dict(zip(frame["key"], frame["text"], strict=True))
    assert texts["c_g1_0"] == "w Taa Tzz"   # sorted by anchor id: aa before zz


def test_build_frame_reads_a_custom_corpus_with_no_LOCUS_DATA_DIR(
    monkeypatch, tmp_path, second_corpus
):
    """The second place `load_papers` is called, and the second dead end.

    `build.pipeline --corpus` alone was not enough: this module also calls
    `load_papers`, so a custom corpus that built and pooled still had no way
    to reach embeddings -- and therefore none to reach `eval` or `verify`.
    Runs the real build on the committed second corpus first, so the
    pickles this reads are the ones that path actually produces.
    """
    from locus.build import pipeline as build_day1

    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    monkeypatch.delenv("LOCUS_DATA_DIR", raising=False)
    assert build_day1.main(
        ["--corpus", str(second_corpus), "--split", "test", "--pool-size", "7"]
    ) == 0

    frame, manifest = build_frame("test", corpus=str(second_corpus))

    assert manifest["corpus"] == str(second_corpus)
    assert manifest["rows"] == frame.height > 0
    assert set(frame["kind"].unique()) == {"paper", "context"}
    assert manifest["papers_without_title"] == 0
    # Every paper row is title [SEP] abstract, taken from papers.jsonl and not
    # from the upstream corpus that is not on this machine.
    papers = frame.filter(pl.col("kind") == "paper")
    assert all("[SEP]" in t for t in papers["text"].to_list())
    assert any("Synthetic" in t for t in papers["text"].to_list())
