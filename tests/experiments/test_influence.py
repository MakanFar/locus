import pytest
from scipy import sparse

from locus.core.rescore import zscore_pool
from locus.experiments.influence import asymmetry, influence_rows
from locus.scoring.cocitation import PPMI


def _ppmi(pairs, vocab):
    index = {w: i for i, w in enumerate(vocab)}
    m = sparse.lil_matrix((len(vocab), len(vocab)))
    for (a, b), v in pairs.items():
        m[index[a], index[b]] = v
        m[index[b], index[a]] = v
    return PPMI(matrix=m.tocsr(), index=index)


def _pool(base, anchors, gold="g", cid="c", pid="p"):
    return {"context_id": cid, "citing_id": pid, "gold": gold,
            "base": base, "z": zscore_pool(base),
            "anchors": frozenset(anchors)}


VOCAB = ["g", "d1", "d2", "a1", "a2", "a3"]


def test_score_influence_matches_the_closed_form():
    # lam * (p_a - mean_A) / (|A| - 1). If influence_rows recomputed the mean
    # over the FULL set when scoring the reduced one -- the easy bug -- every
    # d_score would come out zero and this would fail on all three anchors.
    lam = 3.0
    ppmi = _ppmi({("g", "a1"): 5.0, ("g", "a2"): 2.0, ("g", "a3"): 0.5}, VOCAB)
    pool = _pool({"g": 0.5, "d1": 0.4, "d2": 0.3}, ["a1", "a2", "a3"])
    rows = {r["anchor"]: r for r in influence_rows(pool, ppmi, lam)}
    n = 3
    mean = (5.0 + 2.0 + 0.5) / n
    for a, p_a in (("a1", 5.0), ("a2", 2.0), ("a3", 0.5)):
        assert rows[a]["d_score"] == pytest.approx(
            lam * (p_a - mean) / (n - 1))


def test_an_average_anchor_has_zero_score_influence():
    # The mean normalisation makes influence relative. Three anchors with
    # IDENTICAL association to the target each contribute nothing on removal,
    # however strong that association is. A formula missing the mean
    # subtraction would give each of them lam * p_a / (n-1) instead.
    ppmi = _ppmi({("g", "a1"): 9.0, ("g", "a2"): 9.0, ("g", "a3"): 9.0}, VOCAB)
    pool = _pool({"g": 0.5, "d1": 0.4, "d2": 0.3}, ["a1", "a2", "a3"])
    for r in influence_rows(pool, ppmi, 3.0):
        assert r["d_score"] == pytest.approx(0.0)


def test_single_anchor_influence_is_the_whole_term():
    # |A| = 1 removes to the empty set, where mean_over returns 0.0, so the
    # influence is the entire lam * p_a and the (n-1) closed form does not
    # apply. Dividing by n-1 here would be a ZeroDivisionError.
    ppmi = _ppmi({("g", "a1"): 4.0}, VOCAB)
    pool = _pool({"g": 0.5, "d1": 0.4, "d2": 0.3}, ["a1"])
    (row,) = influence_rows(pool, ppmi, 2.0)
    assert row["n_anchors"] == 1
    assert row["d_score"] == pytest.approx(2.0 * 4.0)


def test_rank_influence_can_oppose_the_targets_own_score():
    # The point of measuring rank separately. a1 is strongly co-cited with the
    # target (PPMI 10) but far more strongly with the distractor (100), so
    # holding it RAISES the target's score and LOSES it the top rank.
    # Removing a1 therefore costs score (d_score > 0) and gains rank
    # (d_rr < 0). Any d_rr derived from the target's own score alone is
    # forced to agree in sign and can never produce this row.
    ppmi = _ppmi({("g", "a1"): 10.0, ("d1", "a1"): 100.0}, VOCAB)
    pool = _pool({"g": 1.0, "d1": 0.9, "d2": 0.1}, ["a1", "a2"])
    rows = {r["anchor"]: r for r in influence_rows(pool, ppmi, 1.0)}
    assert rows["a1"]["d_score"] > 0
    assert rows["a1"]["d_rr"] < 0


def test_rr_and_hit_influence_use_the_full_rescored_pool():
    # a1 carries the target from second place to first. Removing it must show
    # up in BOTH d_rr and d_hit.
    ppmi = _ppmi({("g", "a1"): 20.0}, VOCAB)
    pool = _pool({"g": 0.4, "d1": 0.9, "d2": 0.1}, ["a1", "a2"])
    rows = {r["anchor"]: r for r in influence_rows(pool, ppmi, 1.0)}
    assert rows["a1"]["d_rr"] == pytest.approx(0.5)   # 1.0 -> 0.5
    assert rows["a1"]["d_hit"] == pytest.approx(1.0)  # 1.0 -> 0.0


def test_no_anchors_yields_no_interventions():
    ppmi = _ppmi({}, VOCAB)
    assert influence_rows(_pool({"g": 1.0, "d1": 0.5}, []), ppmi, 1.0) == []


def test_asymmetry_pairs_only_within_one_citing_paper():
    # Two different papers each citing {x, y} must not be matched into a
    # reciprocal pair: their pools and anchor sets are unrelated.
    rows = [
        {"citing_id": "p1", "gold": "x", "anchor": "y", "d_rr": 0.4},
        {"citing_id": "p2", "gold": "y", "anchor": "x", "d_rr": -0.4},
    ]
    assert asymmetry(rows)["pairs"] == 0


def test_asymmetry_matches_the_two_directions_of_one_pair():
    rows = [
        {"citing_id": "p1", "gold": "x", "anchor": "y", "d_rr": 0.4},
        {"citing_id": "p1", "gold": "y", "anchor": "x", "d_rr": -0.2},
    ]
    got = asymmetry(rows)
    assert got["pairs"] == 1
    assert got["mean_abs_difference"] == pytest.approx(0.6)
    assert got["frac_same_sign"] == pytest.approx(0.0)
