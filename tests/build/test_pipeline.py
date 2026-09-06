import json

import pytest

from locus import config
from locus.build import pipeline as build_day1
from locus.data.corpus import build_index
from locus.data.ingest import jsonl_papers, jsonl_source


def test_main_fails_fast_on_empty_index(monkeypatch, tmp_path):
    # No real dataset access: build_index is monkeypatched to return an empty
    # index, so nothing is streamed from contexts.json or papers.json.
    monkeypatch.setattr(build_day1, "build_index", lambda split, **_: {})
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    # This test is about the empty-index guard, not the LOCUS_DATA_DIR guard,
    # so it must not depend on the developer's shell having the variable set:
    # CI runs pytest with no such variable, and an ambient LOCUS_DATA_DIR
    # would let this test pass by accident while masking a real regression.
    monkeypatch.setenv("LOCUS_DATA_DIR", str(tmp_path))

    # argv=[] explicitly: main() now parses arguments, and the default
    # argv=None would make argparse read pytest's own sys.argv and exit 2
    # before ever reaching the empty-index guard this test is about.
    with pytest.raises(SystemExit, match="0 contexts loaded"):
        build_day1.main([])


def test_empty_index_message_survives_a_missing_data_dir(monkeypatch, tmp_path):
    # With LOCUS_DATA_DIR set (the test above), config.contexts() never
    # raises, so the guard's `except SystemExit` fallback is never exercised
    # there. This is the only case that pins it: with the variable unset,
    # config.contexts() itself used to raise while the guard was building its
    # message, replacing the "0 contexts loaded" diagnosis with a misleading
    # "LOCUS_DATA_DIR is not set" -- losing the very error this guard exists
    # to report. Without the try/except in build/pipeline.py, this test fails.
    monkeypatch.setattr(build_day1, "build_index", lambda split, **_: {})
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    monkeypatch.delenv("LOCUS_DATA_DIR", raising=False)

    with pytest.raises(SystemExit, match="0 contexts loaded"):
        build_day1.main([])


def test_the_whole_build_runs_on_a_custom_corpus_with_no_LOCUS_DATA_DIR(
    monkeypatch, tmp_path, second_corpus
):
    """The entry point a reviewer with their own corpus actually calls.

    `LOCUS_DATA_DIR` is deleted first, because that is the failure this flag
    exists to fix: `--corpus` used to redirect the CONTEXTS pass only, so the
    build got as far as step 4 and exited with "LOCUS_DATA_DIR is not set"
    from `load_papers`. Running `main` rather than the library functions is
    the point -- the library path was already covered and still dead-ended
    here.
    """
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    monkeypatch.delenv("LOCUS_DATA_DIR", raising=False)

    # 7, not the published 10: this corpus has 4 locations of 2 references per
    # citing paper, so 6 cross-location distractors are all it can offer.
    rc = build_day1.main(
        ["--corpus", str(second_corpus), "--split", "test", "--pool-size", "7"]
    )

    assert rc == 0, "the custom-corpus build did not reach a clean exit"
    report = json.loads((tmp_path / "build_report_test.json").read_text())
    assert report["corpus"] == str(second_corpus)
    assert report["config"]["pool_size"] == 7
    assert report["rank_items"] > 0
    assert report["hard"] > 0
    assert report["headline"] > 0
    assert report["failures"] == []
    # The two Gu-calibrated thresholds are reported, not enforced: a corpus of
    # 64 rank items cannot have a 30,000-item headline set.
    assert any("headline set" in note for note in report["scale_notes"])
    for name in (
        "index_test.pkl", "rank_items_test.pkl", "swap_pairs_test.pkl",
        "hard_ids_test.pkl", "headline_ids_test.pkl",
    ):
        assert (tmp_path / name).exists(), f"{name} was not written"


def test_the_published_build_still_enforces_the_scale_gates(
    monkeypatch, tmp_path, second_corpus
):
    """`--corpus` is the only thing that softens them.

    Same corpus, reached through `--pool-size` alone: without `--corpus` the
    headline and hard-fraction thresholds are hard failures, exactly as they
    were before the flag existed. This is what stops the note mechanism from
    quietly becoming a way to pass a failing published build.
    """
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    monkeypatch.setattr(
        build_day1, "build_index",
        lambda split, **_: build_index(split, source=jsonl_source(second_corpus)),
    )
    monkeypatch.setattr(
        build_day1, "load_papers",
        lambda ids, **_: jsonl_papers(second_corpus)(ids),
    )

    rc = build_day1.main(["--split", "test", "--pool-size", "7"])

    assert rc == 1
    report = json.loads((tmp_path / "build_report_test.json").read_text())
    assert report["scale_notes"] == []
    assert any("headline set" in f for f in report["failures"])
    assert any("hard fraction" in f for f in report["failures"])
