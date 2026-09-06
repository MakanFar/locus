"""Where a corpus's raw rows come from, separated from what is built out of them.

`corpus._build_recs` holds every piece of logic that matters -- location
clustering, marker alignment, anchor recovery -- and must be shared by every
corpus rather than reimplemented per format. This module supplies only the rows
it consumes: dicts with `context_id`, `citing_id`, `refid`, `raw`, `masked`.

All five are required. `raw` is not decorative: `cluster()` assigns the location
label from it, and `markers_with_numbers(raw, masked)` aligns numbered markers
against masked positions, which is the entire marker evidence route.

A corpus has a second side the harness reads -- the cited papers' metadata,
for hardness stratification and for the text that gets embedded -- so this
module carries a second protocol, `PaperSource`, for exactly the same reason
`RecordSource` exists: `corpus.load_papers` used to reach straight into
`config.papers()`, which meant a corpus that supplied its own contexts still
died at the papers pass with "LOCUS_DATA_DIR is not set". The two protocols
are deliberately symmetric: one supplies rows, one supplies papers, and
nothing downstream of either can tell where they came from.
"""
from __future__ import annotations

import collections
from collections.abc import Iterator
from typing import Protocol

REQUIRED = ("context_id", "citing_id", "refid", "raw", "masked")

# What `load_papers` guarantees its callers, whatever the source.
PAPER_FIELDS = ("title", "abstract", "authors")


class RecordSource(Protocol):
    """Yields one corpus row per context, for the named split."""

    def __call__(self, split: str) -> Iterator[dict]: ...


class PaperSource(Protocol):
    """Returns `{paper_id: {"title", "abstract", "authors"}}` for `ids`.

    Must raise rather than return a short dict: a silently missing paper
    shrinks the hard and headline slices while every gate still passes.
    """

    def __call__(self, ids: set[str]) -> dict[str, dict]: ...


def grouped(
    source: RecordSource, split: str, *, sort: bool
) -> Iterator[tuple[str, list[dict]]]:
    """Yield (citing_id, rows) one paper at a time.

    `sort=False` streams in file order and refuses a re-opened citing_id --
    correct for the upstream corpus, whose contiguity was verified across all
    3,205,210 records, and where a violation means the file is not what we
    think it is. `sort=True` buffers and groups, which is the only safe reading
    for a file someone else assembled.
    """
    if sort:
        buckets: dict[str, list[dict]] = collections.defaultdict(list)
        for row in source(split):
            _check(row)
            buckets[row["citing_id"]].append(row)
        for citing_id in sorted(buckets):
            yield citing_id, buckets[citing_id]
        return

    closed: set[str] = set()
    current: str | None = None
    rows: list[dict] = []
    for row in source(split):
        _check(row)
        citing_id = row["citing_id"]
        if citing_id != current:
            if rows:
                yield current, rows
            if citing_id in closed:
                raise SystemExit(
                    f"corpus is not contiguous by citing_id: {citing_id!r} "
                    "re-opens after its run closed. Grouping assumes "
                    "contiguity, so a split run would build this paper's "
                    "bibliography from only part of its contexts."
                )
            if current is not None:
                closed.add(current)
            current = citing_id
            rows = []
        rows.append(row)
    if rows:
        yield current, rows


def _check(row: dict) -> None:
    missing = [k for k in REQUIRED if k not in row]
    if missing:
        raise SystemExit(
            f"corpus row {row.get('context_id', '<no context_id>')!r} is "
            f"missing {missing}; every row needs {list(REQUIRED)}"
        )
