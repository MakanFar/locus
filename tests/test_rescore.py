import math

import pytest

from locus.core.probe import reciprocal_rank
from locus.core.rescore import rescore, zscore_pool
from locus.core.types import ContextRec
from locus.scoring.cocitation import count_sites


def _rec(cid, refid, loc):
    return ContextRec(
        context_id=f"{cid}_{refid}_{loc}", citing_id=cid, refid=refid,
        raw="", masked="", location=loc,
    )


def _graph():
    # A-B twice, A-D once, and D-E once so that D's marginal is not carried
    # entirely by A. Without that fourth site B and D each co-occur ONLY with
    # A, which makes PMI(B,A) == PMI(D,A) however different the raw counts are
    # -- PMI measures specificity, not frequency.
    #
    # Marginals: A=3, B=2, D=2, E=1; N=8. At alpha=1,
    #   PPMI(B,A) = log(2*8/(2*3)) = 0.981
    #   PPMI(D,A) = log(1*8/(2*3)) = 0.288
    return count_sites([
        ("P1", [_rec("P1", "A", 0), _rec("P1", "B", 0)]),
        ("P2", [_rec("P2", "A", 0), _rec("P2", "B", 0)]),
        ("P3", [_rec("P3", "A", 0), _rec("P3", "D", 0)]),
        ("P4", [_rec("P4", "D", 0), _rec("P4", "E", 0)]),
    ])


class TestZScorePool:
    def test_centres_and_scales(self):
        z = zscore_pool({"a": 1.0, "b": 2.0, "c": 3.0})
        assert math.fsum(z.values()) == pytest.approx(0.0, abs=1e-12)
        assert z["c"] > z["b"] > z["a"]

    def test_is_rank_preserving(self):
        base = {"a": -5.0, "b": 0.25, "c": 100.0}
        z = zscore_pool(base)
        assert sorted(base, key=base.get) == sorted(z, key=z.get)

    def test_zero_variance_pool_gives_zeros_not_nan(self):
        # A constant base scorer produces this, and probe._band raises on NaN,
        # so a 0/0 here would take down the primary metric.
        z = zscore_pool({k: 7.0 for k in "abcdefghij"})
        assert set(z.values()) == {0.0}

    def test_single_candidate_pool(self):
        assert zscore_pool({"only": 3.0}) == {"only": 0.0}


class TestRescore:
    def test_lambda_zero_preserves_the_base_ranking_exactly(self):
        base = {"A": 0.1, "B": 0.9, "D": 0.5}
        out = rescore(base, frozenset({"B"}), _graph().ppmi(min_count=1), lam=0.0)
        assert reciprocal_rank(out, "B") == reciprocal_rank(base, "B")
        assert sorted(out, key=out.get) == sorted(base, key=base.get)

    def test_a_co_cited_candidate_is_promoted(self):
        # B is co-cited with A twice; D once. With anchor A and a large lambda,
        # B must overtake D despite starting below it.
        base = {"B": 0.0, "D": 1.0}
        out = rescore(base, frozenset({"A"}), _graph().ppmi(min_count=1), lam=50.0)
        assert out["B"] > out["D"]

    def test_empty_anchor_set_reduces_to_the_base_ranking(self):
        base = {"A": 0.1, "B": 0.9, "D": 0.5}
        out = rescore(base, frozenset(), _graph().ppmi(min_count=1), lam=3.0)
        assert sorted(out, key=out.get) == sorted(base, key=base.get)

    def test_zero_variance_base_is_ranked_by_structure_alone(self):
        base = {"B": 4.0, "D": 4.0}
        out = rescore(base, frozenset({"A"}), _graph().ppmi(min_count=1), lam=1.0)
        assert out["B"] > out["D"]

    def test_never_emits_nan(self):
        # Every degenerate combination at once.
        p = _graph().ppmi(min_count=1)
        for base in ({"B": 4.0, "D": 4.0}, {"B": 0.0, "D": 1.0}, {"only": 2.0}):
            for anchors in (frozenset(), frozenset({"A"}), frozenset({"NEVER"})):
                for lam in (0.0, 1.0, 1e6):
                    out = rescore(base, anchors, p, lam)
                    assert all(v == v for v in out.values()), (base, anchors, lam)

    def test_keys_are_preserved(self):
        base = {"A": 0.1, "B": 0.9, "D": 0.5}
        assert set(rescore(base, frozenset({"A"}), _graph().ppmi(min_count=1), 1.0)) == set(base)

    def test_a_candidate_that_is_also_an_anchor_produces_no_nan(self):
        # B is both a candidate in the pool and one of the anchors passed to
        # rescore. count_sites never records self-pairs, so PPMI(B,B) == 0
        # and B gets no self-reward -- it must still be ranked ahead of D by
        # its co-citation with the OTHER anchor, A.
        base = {"B": 0.0, "D": 1.0}
        out = rescore(base, frozenset({"A", "B"}), _graph().ppmi(min_count=1), lam=50.0)
        assert all(v == v for v in out.values())
        assert out["B"] > out["D"]
