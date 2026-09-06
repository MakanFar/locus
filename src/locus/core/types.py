"""Pure domain types over which the benchmark's logic is defined.

`ContextRec` and the functions here are pure over the record: none of them
touches the filesystem, the upstream JSON format, or any Gu-format reader.
That is what lets a future corpus ingester produce `ContextRec`s of its own
without going through `corpus.py`'s loaders.
"""
import collections
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextRec:
    context_id: str
    citing_id: str
    refid: str
    raw: str
    masked: str
    location: int
    alignment_ok: bool = True
    marker_anchors: frozenset[str] = frozenset()
    n_unresolvable: int = 0
    n_implausible: int = 0
    n_self: int = 0


def gold_sets(recs: list[ContextRec]) -> dict[int, set[str]]:
    out: dict[int, set[str]] = collections.defaultdict(set)
    for r in recs:
        out[r.location].add(r.refid)
    return dict(out)


def anchors(rec: ContextRec, golds: dict[int, set[str]]) -> set[str]:
    """A(l) via the clustering route: the location's gold set minus this target."""
    return golds.get(rec.location, set()) - {rec.refid}


def anchors_union(rec: ContextRec, golds: dict[int, set[str]]) -> set[str]:
    """A(l) from both evidence routes.

    Clustering evidence is "another context at this location targets it";
    marker evidence is "an OTHERCIT marker in this window resolves to it".
    Measured on the test split the two are near-disjoint at the margin --
    neither subsumes the other -- so the union is the correct reading of
    "references cited at this location".
    """
    return anchors(rec, golds) | set(rec.marker_anchors)


def clustering_disagreement(rec: ContextRec, golds: dict[int, set[str]]) -> int:
    """Marker-derived anchors that clustering did NOT place at this location.

    A context's own OTHERCIT markers sit inside its own 400-char window, so a
    correct location cluster must contain them. Any that are missing mean the
    cluster was split too finely -- the sharpest available diagnostic for THETA,
    and it needs no ground truth.
    """
    return len(rec.marker_anchors - anchors(rec, golds))
