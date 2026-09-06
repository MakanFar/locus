"""Every invariant gets its own violating fixture and its own test.

A validator that reports "the corpus is invalid" is not usable by someone
assembling a corpus for the first time; each check names the row and the rule.
"""
import json

import pytest

from locus.data.ingest import validate

GOOD = [
    {"context_id": "c1", "citing_id": "p1", "refid": "r1",
     "raw": "we build on [1] and [2] here", "masked_text": "we build on  TARGETCIT  OTHERCIT here"},
    {"context_id": "c2", "citing_id": "p1", "refid": "r2",
     "raw": "we build on [1] and [2] here", "masked_text": "we build on OTHERCIT  TARGETCIT  here"},
]
PAPERS = [{"id": "r1", "title": "A", "abstract": "aa"},
          {"id": "r2", "title": "B", "abstract": "bb"}]


def _write(root, contexts=None, papers=None, splits=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "contexts.jsonl").write_text(
        "\n".join(json.dumps(r) for r in (GOOD if contexts is None else contexts)) + "\n")
    (root / "papers.jsonl").write_text(
        "\n".join(json.dumps(r) for r in (PAPERS if papers is None else papers)) + "\n")
    (root / "splits").mkdir(exist_ok=True)
    s = {"train": [], "val": [], "test": ["c1", "c2"]} if splits is None else splits
    for name, ids in s.items():
        (root / "splits" / f"{name}.json").write_text(json.dumps(ids))
    return root


def test_a_valid_corpus_reports_nothing(tmp_path):
    assert validate(_write(tmp_path / "ok")) == []


def test_duplicate_context_id(tmp_path):
    rows = [*GOOD, dict(GOOD[0])]
    problems = validate(_write(tmp_path / "dup", contexts=rows))
    assert any("c1" in p and "duplicate" in p.lower() for p in problems)


def test_refid_not_in_papers(tmp_path):
    rows = [dict(GOOD[0], refid="missing")]
    problems = validate(_write(tmp_path / "ref", contexts=rows,
                               splits={"train": [], "val": [], "test": ["c1"]}))
    assert any("missing" in p for p in problems)


def test_empty_masked_text(tmp_path):
    rows = [dict(GOOD[0], masked_text="   ")]
    problems = validate(_write(tmp_path / "empty", contexts=rows,
                               splits={"train": [], "val": [], "test": ["c1"]}))
    assert any("masked_text" in p for p in problems)


@pytest.mark.parametrize("masked,why", [
    ("no marker at all", "zero"),
    ("two  TARGETCIT  and  TARGETCIT  here", "two"),
])
def test_target_marker_count_must_be_exactly_one(tmp_path, masked, why):
    rows = [dict(GOOD[0], masked_text=masked)]
    problems = validate(_write(tmp_path / f"tgt_{why}", contexts=rows,
                               splits={"train": [], "val": [], "test": ["c1"]}))
    assert any("TARGETCIT" in p for p in problems)


def test_splits_must_be_disjoint_by_citing_paper(tmp_path):
    """The leakage check. Two contexts of the same citing paper in different
    splits means a model can see one location of a paper at train time and be
    scored on another at test time."""
    problems = validate(_write(tmp_path / "leak",
                               splits={"train": ["c1"], "val": [], "test": ["c2"]}))
    assert any("p1" in p and "split" in p.lower() for p in problems)


def _alignment_rows(n_total, n_unalignable):
    """`n_total` otherwise-valid rows, the first `n_unalignable` of which have
    a `raw` that shares no markers with `masked_text`; the rest are copies of
    a known-alignable pair. `refid` alternates over the two ids PAPERS
    resolves, so this stays valid under every other check regardless of size.
    """
    rows = []
    for i in range(n_total):
        raw = (
            "this text shares no markers with the masked one"
            if i < n_unalignable
            else "we build on [1] and [2] here"
        )
        rows.append({
            "context_id": f"c{i}",
            "citing_id": "p1",
            "refid": "r1" if i % 2 == 0 else "r2",
            "raw": raw,
            "masked_text": "we build on  TARGETCIT  OTHERCIT here",
        })
    return rows


def test_alignment_rate_above_threshold_reports_a_problem(tmp_path):
    """3 of 4 rows (75%) fail to align -- well above the default 50% rate.
    A single unalignable row is the normal, expected result of Gu et al.'s
    windows sometimes being cut mid-marker (`corpus._build_recs` treats it as
    non-fatal); most of the corpus failing to align means `raw` does not
    correspond to `masked_text` at all, which is the one condition this check
    exists to catch, and it must be reported."""
    rows = _alignment_rows(n_total=4, n_unalignable=3)
    ids = [r["context_id"] for r in rows]
    problems = validate(_write(tmp_path / "align_bad", contexts=rows,
                               splits={"train": [], "val": [], "test": ids}))
    assert any("3 of 4" in p and "cannot align" in p for p in problems)


def test_alignment_rate_below_threshold_reports_nothing(tmp_path):
    """1 of 5 rows (20%) fails to align -- comfortably below the default 50%
    rate, and exactly the shape of a real corpus (a small fraction of windows
    cut mid-marker). This is normal and must not be reported."""
    rows = _alignment_rows(n_total=5, n_unalignable=1)
    ids = [r["context_id"] for r in rows]
    problems = validate(_write(tmp_path / "align_ok", contexts=rows,
                               splits={"train": [], "val": [], "test": ids}))
    assert problems == []


def test_contexts_file_absent(tmp_path):
    """`contexts.jsonl` missing entirely. The message must name the missing
    file and say so in words no other check's message uses ("not found"),
    not just any mention of the filename -- check 2's own message also
    contains "contexts.jsonl", so the filename alone would not discriminate."""
    root = tmp_path / "no_contexts"
    root.mkdir()
    (root / "papers.jsonl").write_text(
        "\n".join(json.dumps(r) for r in PAPERS) + "\n")
    (root / "splits").mkdir()
    for name, ids in {"train": [], "val": [], "test": ["c1", "c2"]}.items():
        (root / "splits" / f"{name}.json").write_text(json.dumps(ids))
    problems = validate(root)
    assert any("contexts.jsonl" in p and "not found" in p.lower() for p in problems)


def test_papers_file_absent(tmp_path):
    """`papers.jsonl` missing entirely. Check 3's own message also contains
    "papers.jsonl" (`"...refid ... is not in papers.jsonl"`) but never the
    words "not found", so requiring both together cannot be satisfied by
    check 3 firing on some other row."""
    root = tmp_path / "no_papers"
    root.mkdir()
    (root / "contexts.jsonl").write_text(
        "\n".join(json.dumps(r) for r in GOOD) + "\n")
    (root / "splits").mkdir()
    for name, ids in {"train": [], "val": [], "test": ["c1", "c2"]}.items():
        (root / "splits" / f"{name}.json").write_text(json.dumps(ids))
    problems = validate(root)
    assert any("papers.jsonl" in p and "not found" in p.lower() for p in problems)


def test_splits_directory_absent(tmp_path):
    """The whole `splits/` directory missing. The message must name the
    specific split file ("splits/train.json"), a path no other check ever
    emits -- check 6's leakage message says only the bare word "splits", so
    the full relative path is what makes this assertion check-1-specific."""
    root = tmp_path / "no_splits"
    root.mkdir()
    (root / "contexts.jsonl").write_text(
        "\n".join(json.dumps(r) for r in GOOD) + "\n")
    (root / "papers.jsonl").write_text(
        "\n".join(json.dumps(r) for r in PAPERS) + "\n")
    problems = validate(root)
    assert any(
        "splits/train.json" in p and "not found" in p.lower() for p in problems
    )


def test_malformed_line_in_contexts_names_the_line(tmp_path):
    """A syntactically broken line in an otherwise well-formed
    `contexts.jsonl`. On a million-row corpus, "contexts.jsonl is invalid"
    tells the user nothing; the message must give the line number, which no
    other check's message ever contains."""
    root = tmp_path / "malformed"
    root.mkdir()
    (root / "contexts.jsonl").write_text(
        json.dumps(GOOD[0]) + "\n" + "{this is not valid json" + "\n")
    (root / "papers.jsonl").write_text(
        "\n".join(json.dumps(r) for r in PAPERS) + "\n")
    (root / "splits").mkdir()
    for name, ids in {"train": [], "val": [], "test": ["c1"]}.items():
        (root / "splits" / f"{name}.json").write_text(json.dumps(ids))
    problems = validate(root)
    assert any("contexts.jsonl" in p and "line 2" in p for p in problems)
