import numpy as np
from scipy import sparse

from locus.eval.controls import (
    degree_bins,
    degree_matched_anchors,
    random_anchors,
)
from locus.scoring.cocitation import CoGraph


def _graph(marginals):
    vocab = tuple(f"p{i}" for i in range(len(marginals)))
    n = len(vocab)
    return CoGraph(
        counts=sparse.csr_matrix((n, n)),
        marginals=np.asarray(marginals, dtype=np.int64),
        n_total=int(sum(marginals)),
        vocab=vocab,
        index={w: i for i, w in enumerate(vocab)},
    )


def _pool(cid, anchors):
    return {"context_id": cid, "citing_id": "c", "gold": "g",
            "base": {}, "z": {}, "anchors": frozenset(anchors)}


def test_random_anchors_preserve_set_size():
    # Size matters: mean_over divides by |A|, so a control with a different
    # anchor count would change the scale of the PPMI term as well as its
    # content, and the comparison would confound the two.
    g = _graph([1] * 50)
    pools = [_pool("a", {"p1"}), _pool("b", {"p1", "p2", "p3"}), _pool("c", set())]
    got = random_anchors(pools, list(g.vocab), seed=0)
    assert [len(x) for x in got] == [1, 3, 0]


def test_random_anchors_are_deterministic_and_context_specific():
    g = _graph([1] * 50)
    pools = [_pool("a", {"p1"}), _pool("b", {"p1"})]
    first = random_anchors(pools, list(g.vocab), seed=0)
    assert first == random_anchors(pools, list(g.vocab), seed=0)
    assert first != random_anchors(pools, list(g.vocab), seed=1)
    # Two contexts with identical true anchors must not get identical draws,
    # or the control collapses toward a single shared anchor set.
    assert first[0] != first[1]


def test_degree_bins_group_by_log2_marginal():
    g = _graph([1, 1, 2, 3, 4, 7, 8, 100])
    of_paper, members, keys = degree_bins(g)
    assert of_paper["p0"] == 0          # log2(1) -> 0
    assert of_paper["p2"] == 1          # log2(2) -> 1
    assert of_paper["p3"] == 1          # log2(3) -> 1
    assert of_paper["p4"] == 2          # log2(4) -> 2
    assert of_paper["p7"] == 6          # log2(100) -> 6
    assert sum(len(m) for m in members) == len(g.vocab)
    assert keys == sorted(keys)


def test_degree_matched_draws_from_the_same_bin():
    # A high-degree anchor has nonzero PPMI with far more candidates, so a
    # uniform random control is systematically LOW degree and easier to beat
    # than it looks. Matching the bin removes that advantage.
    marginals = [1] * 20 + [1000] * 20
    g = _graph(marginals)
    of_paper, _, _ = degree_bins(g)
    high = [w for w in g.vocab if of_paper[w] == of_paper["p20"]]
    pools = [_pool("a", {"p20"})]          # a high-degree true anchor
    got = degree_matched_anchors(pools, g, seed=0)
    (replacement,) = got[0]
    assert replacement in high
    assert of_paper[replacement] == of_paper["p20"]


def test_degree_matched_preserves_size_and_is_deterministic():
    g = _graph([1] * 10 + [64] * 10)
    pools = [_pool("a", {"p0", "p11"})]
    got = degree_matched_anchors(pools, g, seed=0)
    assert len(got[0]) == 2
    assert got == degree_matched_anchors(pools, g, seed=0)


def test_anchor_outside_the_graph_still_gets_a_replacement():
    # A true anchor absent from the train-only vocabulary has no bin; it must
    # still be replaced, or the control would silently shrink the anchor set.
    g = _graph([1] * 10)
    pools = [_pool("a", {"not_in_graph"})]
    got = degree_matched_anchors(pools, g, seed=0)
    assert len(got[0]) == 1
    assert next(iter(got[0])) in g.vocab


class _TablePPMI:
    def __init__(self, table):
        self.table = table

    def mean_over(self, c, anchors):
        if not anchors:
            return 0.0
        return sum(self.table.get((c, a), 0.0) for a in anchors) / len(anchors)


class _FakeGraph:
    """Just enough CoGraph for controls.main: vocabulary, degrees, a PPMI view."""

    def __init__(self, ppmi, vocab):
        self._ppmi = ppmi
        self.vocab = tuple(vocab)
        self.marginals = np.ones(len(vocab), dtype=np.int64)

    def ppmi(self, alpha, min_count):
        return self._ppmi


def test_main_reports_every_arm_against_the_base(monkeypatch, tmp_path):
    # The paper's controls table states random and degree-matched anchors
    # against the BASE (0.478 -> 0.477, -> 0.472). The gate pairs them against
    # the true anchors instead, so each arm needs its own base-paired interval
    # in the report or those rows have no CI anywhere.
    import json

    from locus import config
    from locus.eval import controls

    ppmi = _TablePPMI({("g", "a"): 5.0})
    vocab = ["a", "b", "c", "d", "e", "f"]
    pools = [
        {"context_id": "c1", "citing_id": "p1", "gold": "g",
         "base": {"g": 1.0, "x": 2.0}, "z": {"g": 1.0, "x": 2.0},
         "anchors": frozenset(["a"])},
        {"context_id": "c2", "citing_id": "p2", "gold": "g",
         "base": {"g": 2.0, "x": 1.0}, "z": {"g": 2.0, "x": 1.0},
         "anchors": frozenset(["b"])},
    ]
    monkeypatch.setattr(controls, "build_pools", lambda *a, **k: pools)
    monkeypatch.setattr(controls, "base_scorer", lambda *a, **k: None)
    monkeypatch.setattr(controls.CoGraph, "load",
                        classmethod(lambda cls, p: _FakeGraph(ppmi, vocab)))
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    sweep = tmp_path / "sweep.json"
    sweep.write_text(json.dumps({"best": {"alpha": 0.5, "min_count": 1,
                                          "lambda": 1.0}}))
    controls.main(["--embeddings", "x.parquet", "--sweep", str(sweep),
                   "--graph", "g.npz", "--tag", "t"])
    report = json.loads((tmp_path / "controls_t.json").read_text())

    assert report["base_mrr"] == 0.75                    # (0.5 + 1.0) / 2
    assert report["true_vs_base"]["delta"] == 0.25       # gold lifted in pool 1
    for arm in ("random", "degree_matched"):
        row = report["arms"][arm]
        assert {"mrr", "vs_base"} <= set(row)
        assert {"delta", "ci"} <= set(row["vs_base"])
        assert -1.0 <= row["vs_base"]["delta"] <= 1.0
