# src/locus/data/corpus.py
"""Build and freeze the per-citing-paper index the whole harness reads.

One streaming pass over contexts.json, one over papers.json. Everything
downstream (pools, hardness, metrics) works off the cached result.
"""
import pickle
from collections.abc import Iterator
from pathlib import Path

from locus import config
from locus.core.alignment import (
    build_bibliography,
    markers_with_numbers,
    resolve,
)
from locus.core.locations import cluster
from locus.core.types import ContextRec
from locus.data.records import PaperSource, RecordSource, grouped
from locus.data.streaming import stream_dict, stream_list


def _split_context_ids(split: str) -> set[str]:
    # stream_list, not json.load: train.json names 2,988,030 contexts and
    # json.load would build every record before we take the ids.
    return {e["context_id"] for e in stream_list(config.split_file(split))}


def _build_recs(
    citing_id: str,
    rows: list[dict],
    keep_text: bool,
    theta: float,
    shingle_n: int,
) -> list[ContextRec]:
    rows.sort(key=lambda r: r["context_id"])  # determinism
    labels = cluster([r["raw"] for r in rows], theta=theta, n=shingle_n)

    # Two passes: markers first, so the bibliography is complete before any
    # context is validated against it.
    marks = [markers_with_numbers(r["raw"], r["masked"]) for r in rows]
    bib = build_bibliography(
        [(r["refid"], m) for r, m in zip(rows, marks, strict=True) if m is not None]
    )

    recs = []
    for r, lab, m in zip(rows, labels, marks, strict=True):
        raw = r["raw"] if keep_text else ""
        masked = r["masked"] if keep_text else ""
        if m is None:
            recs.append(
                ContextRec(
                    context_id=r["context_id"],
                    citing_id=citing_id,
                    refid=r["refid"],
                    raw=raw,
                    masked=masked,
                    location=lab,
                    alignment_ok=False,
                )
            )
            continue
        res = resolve(m, bib, r["refid"])
        recs.append(
            ContextRec(
                context_id=r["context_id"],
                citing_id=citing_id,
                refid=r["refid"],
                raw=raw,
                masked=masked,
                location=lab,
                alignment_ok=res.target_valid,
                marker_anchors=res.anchors,
                n_unresolvable=res.n_unresolvable,
                n_implausible=res.n_implausible,
                n_self=res.n_self,
            )
        )
    return recs


def iter_papers(
    split: str,
    *,
    source: RecordSource | None = None,
    keep_text: bool = True,
    theta: float = config.THETA,
    shingle_n: int = config.SHINGLE_N,
) -> Iterator[tuple[str, list[ContextRec]]]:
    """Yield one citing paper's records at a time, in file order.

    With no `source`, this reads the upstream contexts.json, which is
    contiguous by citing_id -- verified across all 3,205,210 records: 395,536
    papers, 395,536 runs, zero re-opens -- so a paper can be closed and
    released as soon as the next one opens. That is what keeps the train pass
    flat; materialising it is what costs ~3 GB. Contiguity is a property of
    today's file, not a guarantee, so it is asserted on every run: a split run
    would build that paper's bibliography from only part of its contexts and
    emit plausible-looking wrong anchors.

    A supplied `source` has no reason to arrive grouped, so its rows are
    sorted by citing_id instead of trusting file order.
    """
    if source is None:
        want = _split_context_ids(split)
        found = 0

        def _upstream(_split: str) -> Iterator[dict]:
            nonlocal found
            for key, val in stream_dict(config.contexts()):
                if key not in want:
                    continue
                found += 1
                yield {
                    "context_id": key,
                    "citing_id": str(val["citing_id"]),
                    "refid": str(val["refid"]),
                    "raw": val["raw"],
                    "masked": val["masked_text"],
                }

        for citing_id, rows in grouped(_upstream, split, sort=False):
            yield citing_id, _build_recs(citing_id, rows, keep_text, theta, shingle_n)

        # Only reached on a full pass. A caller that stops early has not shown
        # the file is truncated, so partial consumption must not assert
        # completeness. This has no meaning for a supplied source.
        if found != len(want):
            raise SystemExit(
                f"streamed {found} contexts for split {split!r} but the split file "
                f"names {len(want)} -- {config.contexts()} looks truncated or "
                "otherwise incomplete; stream_dict returns silently at EOF or on "
                "an unterminated record, so this must be checked explicitly"
            )
        return

    for citing_id, rows in grouped(source, split, sort=True):
        yield citing_id, _build_recs(citing_id, rows, keep_text, theta, shingle_n)


def build_index(
    split: str,
    theta: float = config.THETA,
    shingle_n: int = config.SHINGLE_N,
    *,
    source: RecordSource | None = None,
) -> dict[str, list[ContextRec]]:
    """Materialise a whole split. Correct for test (0.10 GB); use iter_papers
    for train, which would be ~3 GB."""
    return dict(iter_papers(split, source=source, theta=theta, shingle_n=shingle_n))


def save_index(index: dict[str, list[ContextRec]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=5)


def load_index(path: str | Path) -> dict[str, list[ContextRec]]:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_papers(
    ids: set[str], *, source: PaperSource | None = None
) -> dict[str, dict]:
    """`{paper_id: {"title", "abstract", "authors"}}` for every id in `ids`.

    `source` is symmetric with `iter_papers`/`build_index`'s: with none, this
    streams the upstream `papers.json` at `config.papers()`; with one, it
    delegates entirely (see `ingest.jsonl_papers`). Without that parameter a
    custom corpus could be indexed and pooled and then died here with
    "LOCUS_DATA_DIR is not set" -- the papers pass was the one step of the
    pipeline that had no way to be told where a corpus lives.
    """
    if source is not None:
        return source(ids)
    out: dict[str, dict] = {}
    for key, val in stream_dict(config.papers()):
        if key in ids:
            out[key] = {
                "title": val.get("title") or "",
                "abstract": val.get("abstract") or "",
                "authors": val.get("authors") or [],
            }
    if len(out) != len(ids):
        # stream_dict returns silently at EOF, so a truncated or incomplete
        # papers.json would otherwise shrink hard_ids and headline_ids while
        # the hard-fraction gate still passed. build_index guards the same
        # failure mode for contexts.json.
        missing = sorted(ids - set(out))
        raise SystemExit(
            f"papers.json is missing {len(missing)} of {len(ids)} requested "
            f"papers, e.g. {missing[:5]}"
        )
    return out
