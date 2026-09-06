"""A record source yields raw rows; grouping and building are shared.

The upstream file happens to be contiguous by citing_id and `iter_papers`
asserts it. A corpus someone else assembled has no reason to be, so the
generic path must SORT instead -- while the upstream path keeps its assertion,
because there a violation means the file is not what we think it is.
"""
import pytest

from locus.data.records import grouped

ROWS = [
    {"context_id": "c3", "citing_id": "p2", "refid": "r1", "raw": "x", "masked": "x"},
    {"context_id": "c1", "citing_id": "p1", "refid": "r2", "raw": "y", "masked": "y"},
    {"context_id": "c2", "citing_id": "p2", "refid": "r3", "raw": "z", "masked": "z"},
    {"context_id": "c4", "citing_id": "p1", "refid": "r4", "raw": "w", "masked": "w"},
]


def test_grouped_sorts_when_asked():
    """Interleaved citing_ids must still produce one group per paper."""
    out = dict(grouped(lambda _split: iter(ROWS), "test", sort=True))
    assert set(out) == {"p1", "p2"}
    assert [r["context_id"] for r in out["p1"]] == ["c1", "c4"]
    assert [r["context_id"] for r in out["p2"]] == ["c3", "c2"]


def test_grouped_rejects_a_reopened_run_when_not_sorting():
    """Without sorting, a re-opened citing_id means the caller's assumption
    about the file is wrong, and a split run would build that paper's
    bibliography from only part of its contexts."""
    with pytest.raises(SystemExit, match="not contiguous"):
        list(grouped(lambda _split: iter(ROWS), "test", sort=False))
