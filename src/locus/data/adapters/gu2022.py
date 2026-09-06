"""Adapter from Gu et al.'s upstream corpus to the JSONL contract.

The upstream corpus is `contexts.json`/`papers.json` plus `train.json`,
`val.json`, `test.json` (see `locus.config.data_dir`'s docstring for the
exact shape). This is by definition valid LOCUS input -- it is what the
published benchmark was built from -- so `convert` followed by
`locus.data.ingest.validate` is the demonstration that the JSONL contract in
`ingest.py` is sufficient, not merely designed: whatever the validator
rejects here names a rule that is too strict, not a defect in this data.

`contexts.json` and `papers.json` run to gigabytes, so both are walked with
`stream_dict` and never `json.load`ed. The split files are streamed too,
with `stream_list`, for the same reason `iter_papers` uses it in corpus.py:
`train.json` alone names millions of records.
"""
from __future__ import annotations

import json
from pathlib import Path

from locus.data.ingest import CONTEXTS_FILE, PAPERS_FILE, SPLITS, SPLITS_DIR
from locus.data.streaming import stream_dict, stream_list


def _split_ids(path: Path) -> list[str]:
    """Read one split file's context ids, in file order.

    Upstream train/val/test.json are arrays of `{"context_id",
    "positive_ids"}` records; the contract's own splits files -- and this
    module's own unit-test fixtures -- are plain arrays of id strings. Both
    element shapes are accepted so the same code reads either without the
    caller having to know which format it is looking at.
    """
    return [e["context_id"] if isinstance(e, dict) else e for e in stream_list(path)]


def convert(
    src: str | Path,
    out: str | Path,
    *,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict:
    """Convert the upstream corpus at `src` into the JSONL contract at `out`.

    `splits` restricts which of train/val/test get their context ids pulled
    into `contexts.jsonl` -- the rest still get an (empty) splits file, so
    `validate` always finds all three, but their contexts are left out of
    the conversion. That is what makes converting just the test split cheap:
    the ~2 GB `contexts.json` is still read start to finish (`stream_dict`
    has no index to skip by), but nothing outside the wanted ids is written.

    `papers.jsonl` is always converted from the whole of `papers.json`,
    regardless of `splits`: a paper can be the refid of contexts in any
    split, and the cost of converting it is one more full streaming pass,
    not one proportional to which splits were asked for.

    Returns a manifest: counts of contexts and papers written, and of ids
    named by each split file.
    """
    src = Path(src)
    out = Path(out)
    (out / SPLITS_DIR).mkdir(parents=True, exist_ok=True)

    manifest: dict = {"splits": {}}
    want: set[str] = set()
    for name in SPLITS:
        ids = _split_ids(src / f"{name}.json") if name in splits else []
        want.update(ids)
        manifest["splits"][name] = len(ids)
        (out / SPLITS_DIR / f"{name}.json").write_text(json.dumps(ids))

    n_contexts = 0
    with open(out / CONTEXTS_FILE, "w", encoding="utf-8") as f:
        for context_id, val in stream_dict(src / "contexts.json"):
            if context_id not in want:
                continue
            f.write(
                json.dumps(
                    {
                        "context_id": context_id,
                        "citing_id": str(val["citing_id"]),
                        "refid": str(val["refid"]),
                        "raw": val["raw"],
                        "masked_text": val["masked_text"],
                    }
                )
                + "\n"
            )
            n_contexts += 1
    manifest["contexts"] = n_contexts

    n_papers = 0
    with open(out / PAPERS_FILE, "w", encoding="utf-8") as f:
        for paper_id, val in stream_dict(src / "papers.json"):
            f.write(
                json.dumps(
                    {
                        "id": paper_id,
                        "title": val.get("title") or "",
                        "abstract": val.get("abstract") or "",
                        # Optional in the contract, but dropping it here would
                        # make the round trip lossy in a way that MOVES the
                        # benchmark: `hardness.is_easy`'s first-author-surname
                        # route is worth 2,239 of 55,488 test rank items (hard
                        # 78.16% -> 82.20%), so a converted corpus without it
                        # would not reproduce the hard fraction it was
                        # extracted from. Upstream stores a JSON array; a
                        # non-list is normalised away rather than passed on,
                        # because `is_easy` refuses one and `validate` reports
                        # it.
                        "authors": val.get("authors")
                        if isinstance(val.get("authors"), list) else [],
                    }
                )
                + "\n"
            )
            n_papers += 1
    manifest["papers"] = n_papers

    return manifest
