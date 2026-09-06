"""The whole pipeline, on a corpus that is not Gu et al.'s.

This is the only test that exercises the custom-corpus path end to end. It is
the difference between a documented contract and a demonstrated one -- see
`tests/data/fixtures/second_corpus/generate.py` for how the corpus itself was
built and why its shape (multiple contexts per location, marker-aligned
raw/masked pairs) is what makes this a real exercise of the machinery rather
than a corpus shaped to make a test pass.

**Why this file's name used to be a promise it did not keep.** It called
itself "the whole pipeline ... end to end" and stopped at `build_rank_items`
-- one call before `load_papers`, which is where the custom-corpus path
actually dead-ended: `load_papers` had no `source` parameter, so hardness
stratification went to `config.papers()` and exited with "LOCUS_DATA_DIR is
not set" for every corpus that was not the upstream one. A test that stops
one call short of the break reports green on a path that does not work, and
the name is what stopped anyone looking. So the run below now goes all the
way through the artefacts a reviewer would actually publish: index, pools,
papers, hardness, the headline slice, and the portable export verified
against its own null controls.
"""
import json
import pickle

import pytest

from locus import config
from locus.core.hardness import is_easy
from locus.core.pools import build_rank_items, build_swap_pairs
from locus.core.probe import reciprocal_rank
from locus.core.types import anchors_union, gold_sets
from locus.data.corpus import build_index, load_papers, save_index
from locus.data.export import export
from locus.data.ingest import jsonl_papers, jsonl_source, validate
from locus.eval.verify import verify_export

# Pool size is a property of the corpus, not a constant: a pool of k needs
# k-1 distractors from OTHER locations in the same citing paper. This corpus
# has 4 locations of 2 references each per citing paper, so every context's
# available cross-location distractor pool is exactly 4*2 - 2 = 6, giving a
# maximum pool of 7 candidates (6 distractors + the gold). That is the
# largest size this corpus supports -- not the published POOL_SIZE=10, which
# needs 9 distractors and this corpus has none to spare.
POOL_SIZE = 7


def test_second_corpus_validates(second_corpus):
    assert validate(second_corpus) == []


def test_pipeline_runs_and_the_floor_is_exact(second_corpus):
    """A constant scorer must land on H_k/k for whatever k this corpus yields.
    The floor moves with pool size, so this asserts the relationship, not the
    published 0.29290 -- that number belongs to a pool of 10."""
    index = build_index("test", source=jsonl_source(second_corpus))
    assert index, "no citing papers were built"
    items, _drops = build_rank_items(index, pool_size=POOL_SIZE, seed=0)
    assert items, "no rank pools were built"
    k = len(items[0].candidates)
    assert all(len(it.candidates) == k for it in items), "mixed pool sizes"
    floor = sum(1.0 / i for i in range(1, k + 1)) / k
    for it in items:
        rr = reciprocal_rank(dict.fromkeys(it.candidates, 1.0), it.gold)
        assert rr == floor, f"tie expectation {rr!r} != floor {floor!r}"


def test_load_papers_reads_the_corpus_and_not_LOCUS_DATA_DIR(
    second_corpus, monkeypatch
):
    """The step the old version of this file stopped one call short of.

    `LOCUS_DATA_DIR` is deleted from the environment first, so this fails the
    way the CLI failed -- `SystemExit: LOCUS_DATA_DIR is not set` -- if
    `load_papers` ever goes back to reaching for the upstream corpus.
    """
    monkeypatch.delenv("LOCUS_DATA_DIR", raising=False)
    index = build_index("test", source=jsonl_source(second_corpus))
    items, _ = build_rank_items(index, pool_size=POOL_SIZE, seed=0)
    papers = load_papers(
        {it.gold for it in items}, source=jsonl_papers(second_corpus)
    )
    assert papers, "no papers were loaded"
    assert set(papers) == {it.gold for it in items}
    for meta in papers.values():
        assert set(meta) == {"title", "abstract", "authors"}
        assert meta["title"]


def test_a_paper_the_corpus_does_not_have_is_an_error_not_a_short_dict(
    second_corpus
):
    """A silently missing paper shrinks the hard and headline slices while
    every gate still passes, which is why the upstream reader raises and this
    one has to as well."""
    source = jsonl_papers(second_corpus)
    with pytest.raises(SystemExit, match="missing 1 of"):
        source({"ref_000_0", "no_such_paper"})


def test_hardness_survives_a_corpus_with_no_authors(second_corpus):
    """`authors` is optional in the contract, and this corpus omits it.

    So `is_easy` must reach title containment rather than raising on a paper
    dict with no `authors` key -- and the hardness pass over the whole corpus
    must complete, which is the thing a KeyError here would have prevented.
    """
    raw = json.loads(
        (second_corpus / "papers.jsonl").read_text().splitlines()[0]
    )
    assert "authors" not in raw, "fixture is meant to omit authors"
    assert not is_easy("some window text", "an unrelated title", None, 0.5)
    assert is_easy("alpha beta gamma", "alpha beta", raw.get("authors"), 0.5)


def test_the_headline_slice_and_the_export_are_reachable(
    second_corpus, monkeypatch, tmp_path
):
    """The end of the pipeline, not the middle of it.

    Runs every stage a reviewer publishing their own benchmark would run --
    index, pools, swap pairs, papers, hardness, the headline slice, the
    portable export -- and finishes on `verify_export`, which recomputes the
    null controls from the exported JSONL alone. That last step is the claim
    the export exists to make, and it had never been made on any corpus but
    Gu et al.'s.
    """
    monkeypatch.delenv("LOCUS_DATA_DIR", raising=False)
    split = "test"
    work = tmp_path / "work"
    work.mkdir()

    index = build_index(split, source=jsonl_source(second_corpus))
    items, _ = build_rank_items(index, pool_size=POOL_SIZE, seed=0)
    pairs = build_swap_pairs(index)
    papers = load_papers(
        {it.gold for it in items}, source=jsonl_papers(second_corpus)
    )

    by_ctx = {r.context_id: r for recs in index.values() for r in recs}
    hard = {
        it.context_id
        for it in items
        if not is_easy(
            by_ctx[it.context_id].raw,
            papers[it.gold]["title"],
            papers[it.gold].get("authors"),
            config.HARDNESS_TAU,
        )
    }
    anchor_ok = set()
    for recs in index.values():
        golds = gold_sets(recs)
        for rec in recs:
            if anchors_union(rec, golds):
                anchor_ok.add(rec.context_id)
    headline = hard & anchor_ok
    assert headline, "the headline slice is empty on this corpus"
    assert headline <= {it.context_id for it in items}

    save_index(index, work / f"index_{split}.pkl")
    for name, obj in (
        (f"rank_items_{split}.pkl", items),
        (f"swap_pairs_{split}.pkl", pairs),
        (f"hard_ids_{split}.pkl", sorted(hard)),
        (f"headline_ids_{split}.pkl", sorted(headline)),
    ):
        with open(work / name, "wb") as f:
            pickle.dump(obj, f, protocol=5)

    out = work / "export"
    manifest = export(split, work, out)
    assert manifest["counts"]["rank_items"] == len(items)
    assert manifest["counts"]["headline"] == len(headline)
    # The floor of the pools actually written, which is this corpus's k and
    # not config.POOL_SIZE -- the manifest describes the file, by design.
    assert manifest["floors"]["pool_size"] == POOL_SIZE
    assert verify_export(split, work, out) == []
