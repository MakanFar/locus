"""The documented JSONL corpus contract, and its validator.

This is the interface a stranger builds their own benchmark against, rather
than the upstream `contexts.json`/`papers.json` pair that `corpus.py` streams
directly:

    <root>/
        contexts.jsonl   {"context_id", "citing_id", "refid", "raw", "masked_text"}
        papers.jsonl     {"id", "title", "abstract", "authors" (optional)}
        splits/{train,val,test}.json    ["context_id", ...]

`jsonl_source` turns rows in this shape into the `RecordSource` that
`corpus.iter_papers`/`build_index` accept, renaming `masked_text` to `masked`
so every row satisfies `records.REQUIRED`. `jsonl_papers` is its other half:
the `PaperSource` that `corpus.load_papers` reads instead of the upstream
`papers.json`. Both are needed, and only having the first is what made the
custom-corpus path dead-end -- a corpus could be indexed and pooled, then
died at hardness stratification with "LOCUS_DATA_DIR is not set".

`authors` is optional, and what omitting it costs is measured rather than
guessed. `core.hardness.is_easy` has two routes: the first author's surname
appearing in the window, and title containment >= tau. Without `authors` only
the second survives, so every context whose window names the author but not
the title moves from easy to hard. On Gu et al.'s test split, over the same
55,488 rank items: hard 43,372 (78.16%) with authors against 45,611 (82.20%)
without -- 2,239 contexts, 4.04 pp -- and the pre-registered headline slice
moves with it, 35,701 to 37,208. That is why `authors` is in the contract at
all: a corpus that
omits it is a valid LOCUS corpus and will build, but it is not the corpus this
benchmark's published hard fraction was measured on, and `validate` says so.

`validate` checks a directory against the contract before anyone hands it to
the harness. A validator that only says "the corpus is invalid" is useless to
someone assembling their first corpus: every problem returned names the
offending id and the rule it broke, and every check runs to completion even
after earlier ones have already found violations.

`raw` earns its place in the contract, not just in `records.REQUIRED`, because
of what checking it catches: `markers_with_numbers(raw, masked)` is the entire
marker-evidence route (70,357 of 78,334 test contexts get an anchor through
it, non-nested with the clustering route). A row that fails to align loses
that route silently -- no error, just a smaller anchor set -- which is why
alignability is a validator check and not merely a documented requirement.

A row failing to align is not, by itself, a defect: it is the normal result
of Gu et al.'s fixed-length windows sometimes being cut mid-marker, and
`corpus._build_recs` already treats it as an expected, non-fatal case --
`alignment_ok=False`, no marker anchors, falls back to the clustering route
-- rather than raising. `build/pipeline.py` counts it and reports it as a
metric for exactly that reason. So the check below is a corpus-level *rate*,
not a per-row error: a corpus where a small fraction of windows got cut
mid-marker is normal (Gu et al.'s own real test split sits at 2.33%), while
one where most rows cannot align means `raw` does not correspond to
`masked_text` at all, and the whole marker-evidence route is silently gone --
that is the failure worth catching.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from locus.core.alignment import markers_with_numbers
from locus.data.records import PaperSource, RecordSource

CONTEXTS_FILE = "contexts.jsonl"
PAPERS_FILE = "papers.jsonl"
SPLITS_DIR = "splits"

SPLITS = ("train", "val", "test")


def jsonl_source(root: str | Path) -> RecordSource:
    """A `RecordSource` reading `<root>/contexts.jsonl`.

    Filters rows to the ids named in `<root>/splits/<split>.json` and renames
    `masked_text` to `masked`, so what it yields matches `records.REQUIRED`
    exactly -- everything downstream (`grouped`, `_build_recs`) is oblivious
    to whether its rows came from here or from the upstream corpus.
    """
    root = Path(root)

    def _source(split: str) -> Iterator[dict]:
        want = set(json.loads((root / SPLITS_DIR / f"{split}.json").read_text()))
        with open(root / CONTEXTS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row["context_id"] not in want:
                    continue
                yield {
                    "context_id": row["context_id"],
                    "citing_id": row["citing_id"],
                    "refid": row["refid"],
                    "raw": row["raw"],
                    "masked": row["masked_text"],
                }

    return _source


def jsonl_papers(root: str | Path) -> PaperSource:
    """A `PaperSource` reading `<root>/papers.jsonl`.

    The other half of `jsonl_source`. `corpus.load_papers` calls this instead
    of streaming the upstream `papers.json`, which is what lets a custom
    corpus reach hardness stratification and `build.embed_inputs` at all.

    Yields the same three fields for every id, whatever the file carries:
    `authors` is optional in the contract and defaults to `[]`, so no consumer
    has to branch on whether it was supplied (`hardness.is_easy` then simply
    has one route instead of two -- see this module's docstring for what that
    costs).

    Missing ids raise, matching the upstream reader exactly: a short return
    would quietly shrink the hard and headline slices while every gate still
    passed.
    """
    root = Path(root)

    def _papers(ids: set[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        with open(root / PAPERS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                pid = row.get("id")
                if pid in ids:
                    out[pid] = {
                        "title": row.get("title") or "",
                        "abstract": row.get("abstract") or "",
                        "authors": row.get("authors") or [],
                    }
        if len(out) != len(ids):
            missing = sorted(ids - set(out))
            raise SystemExit(
                f"{root / PAPERS_FILE} is missing {len(missing)} of {len(ids)} "
                f"requested papers, e.g. {missing[:5]}"
            )
        return out

    return _papers


def _load_jsonl(path: Path, problems: list[str], label: str) -> list[dict] | None:
    if not path.exists():
        problems.append(f"{label}: file not found at {path}")
        return None
    rows: list[dict] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            problems.append(f"{label}: line {lineno} is not valid JSON: {e}")
            return None
    return rows


def _load_split(path: Path, problems: list[str], label: str) -> list[str] | None:
    if not path.exists():
        problems.append(f"{label}: file not found at {path}")
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        problems.append(f"{label}: not valid JSON: {e}")
        return None
    if not isinstance(data, list):
        problems.append(f"{label}: expected a JSON array of context_id strings")
        return None
    return data


def validate(root: str | Path, *, max_unalignable_rate: float = 0.5) -> list[str]:
    """Check `root` against the JSONL corpus contract.

    Returns one human-readable problem per violation, each naming the
    offending id and the rule it broke. An empty list means the corpus is
    ready for `jsonl_source`. Every check below runs and reports all of its
    own violations; nothing stops at the first problem found.

    `max_unalignable_rate` is the one check that reports on the whole corpus
    rather than per row -- see check 8 below. Tighten it with a keyword
    argument if a stricter guarantee is wanted than "not obviously broken".
    """
    root = Path(root)
    problems: list[str] = []

    # 1. Files present and parseable. Nothing past this point can be checked
    # meaningfully against data that failed to load, so a failure here is the
    # one case that ends validation early.
    contexts = _load_jsonl(root / CONTEXTS_FILE, problems, CONTEXTS_FILE)
    papers = _load_jsonl(root / PAPERS_FILE, problems, PAPERS_FILE)
    splits: dict[str, list[str] | None] = {}
    for name in SPLITS:
        label = f"{SPLITS_DIR}/{name}.json"
        splits[name] = _load_split(root / SPLITS_DIR / f"{name}.json", problems, label)

    if contexts is None or papers is None or any(v is None for v in splits.values()):
        return problems

    # 2. context_id unique.
    seen: set[str] = set()
    dupes: set[str] = set()
    for row in contexts:
        cid = row.get("context_id")
        if cid in seen:
            dupes.add(cid)
        seen.add(cid)
    problems.extend(
        f"contexts.jsonl: duplicate context_id {cid!r}" for cid in sorted(dupes)
    )

    # 3. refid resolves in papers.jsonl.
    paper_ids = {p.get("id") for p in papers}
    for row in contexts:
        refid = row.get("refid")
        if refid not in paper_ids:
            problems.append(
                f"context {row.get('context_id')!r}: refid {refid!r} is not "
                "in papers.jsonl"
            )

    # 4. masked_text non-empty after strip.
    problems.extend(
        f"context {row.get('context_id')!r}: masked_text is empty"
        for row in contexts
        if not str(row.get("masked_text", "")).strip()
    )

    # 5. Exactly one TARGETCIT.
    for row in contexts:
        n = str(row.get("masked_text", "")).count("TARGETCIT")
        if n != 1:
            problems.append(
                f"context {row.get('context_id')!r}: expected exactly one "
                f"TARGETCIT marker, found {n}"
            )

    # 6. Splits disjoint by citing_id -- the leakage check. Two contexts of
    # the same citing paper in different splits means a model can see one
    # location of a paper at train time and be scored on another at test
    # time.
    citing_of = {row.get("context_id"): row.get("citing_id") for row in contexts}
    owner: dict[str, str] = {}
    for name in SPLITS:
        for cid in splits[name]:
            citing_id = citing_of.get(cid)
            if citing_id is None:
                continue
            other = owner.get(citing_id)
            if other is not None and other != name:
                problems.append(
                    f"citing paper {citing_id!r} appears in both the {other!r} "
                    f"and {name!r} splits"
                )
            else:
                owner[citing_id] = name

    # 7. authors, where supplied, is a JSON array. Absent is legal -- the
    # field is optional and its absence costs one hardness route, which this
    # module's docstring quantifies. Present but the wrong shape is not: a
    # bare string "Yu, Pengqian" would make `is_easy` iterate its characters
    # and test whether "Y" is a token in the window, which is always false,
    # so the surname route would be silently dead rather than absent.
    problems.extend(
        f"paper {p.get('id')!r}: authors must be a JSON array, got "
        f"{type(p['authors']).__name__}"
        for p in papers
        if "authors" in p and not isinstance(p["authors"], list)
    )

    # 8. raw/masked_text alignment rate -- the marker evidence route. A
    # single unalignable row is expected (Gu et al.'s fixed-length windows
    # sometimes cut a citation marker in half, which markers_with_numbers
    # cannot recover from; corpus._build_recs already treats this as normal:
    # alignment_ok=False, falls back to clustering). What is not normal is
    # most of the corpus failing to align, which means raw does not
    # correspond to masked_text at all -- so this is a rate check over the
    # whole corpus, not a report per offending row.
    unalignable = [
        row
        for row in contexts
        if markers_with_numbers(row.get("raw", ""), row.get("masked_text", ""))
        is None
    ]
    if contexts:
        rate = len(unalignable) / len(contexts)
        if rate > max_unalignable_rate:
            problems.append(
                f"{len(unalignable)} of {len(contexts)} contexts "
                f"({rate:.2%}) have raw/masked_text that markers_with_numbers "
                f"cannot align, above max_unalignable_rate={max_unalignable_rate:.2%}. "
                "A small fraction is normal -- Gu et al.'s own real test "
                "split sits at 2.33%, from windows cut mid-marker -- but a "
                "rate this high means raw does not correspond to "
                "masked_text and the marker-evidence route is effectively "
                "gone."
            )

    return problems
