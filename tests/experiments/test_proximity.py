"""How close an anchor sits to the target marker, and what that is worth.

Three tiers, read off the masked window with no model and no annotation:
`group` (only punctuation, conjunctions or other markers between the two),
`sentence` (no sentence end between them) and `beyond` (a sentence ends
between them). They are a proxy for the hand labels of the annotation study,
so the rules that decide a tier are pinned here one case at a time.
"""
import pytest

from locus.experiments.proximity import (
    TIERS,
    gain_with,
    tier,
    tiers_from_markers,
)


class TestTier:
    @pytest.mark.parametrize("between", ["", " ", " , ", " ; ", " and ", " & ",
                                         " OTHERCIT ", " , OTHERCIT , ", " \u2013 "])
    def test_only_punctuation_conjunctions_or_markers_is_one_group(self, between):
        assert tier(between) == "group"

    def test_content_words_inside_one_sentence(self):
        assert tier(" , optics ") == "sentence"
        assert tier(" while Y degrades it ") == "sentence"

    def test_a_sentence_end_between_the_markers(self):
        assert tier(" . However, the average accuracy drops ") == "beyond"
        assert tier(" ? In practice ") == "beyond"

    def test_a_sentence_that_ends_right_before_the_second_marker(self):
        # "... as in OTHERCIT . TARGETCIT showed ..." -- nothing but a full
        # stop lies between, yet they are two sentences. Calling that a group
        # would put the very closest tier on a pair in different sentences.
        assert tier(" . ") == "beyond"
        assert tier(" was reported . ") == "beyond"

    def test_abbreviations_and_decimals_do_not_end_a_sentence(self):
        assert tier(" et al. showed in Fig. 3 that ") == "sentence"
        assert tier(" with ratio 0.5 as in ") == "sentence"
        assert tier(" i.e. the scheme of ") == "sentence"

    def test_semicolons_do_not_end_a_sentence(self):
        assert tier(" ; in contrast the model of ") == "sentence"


def test_tiers_from_markers_takes_the_closest_occurrence_and_skips_the_target():
    #        0         1          2                        3
    masked = "A TARGETCIT OTHERCIT , optics OTHERCIT here. Later OTHERCIT and OTHERCIT ."
    marks = [("TARGETCIT", "1"), ("OTHERCIT", "2"), ("OTHERCIT", "3"),
             ("OTHERCIT", "4"), ("OTHERCIT", "1")]
    refs = {"1": "own", "2": "p2", "3": "p3", "4": "p4"}
    got = tiers_from_markers(masked, marks, refs, own_refid="own")
    assert got == {"p2": "group", "p3": "sentence", "p4": "beyond"}


def test_tiers_from_markers_is_empty_without_exactly_one_target():
    assert tiers_from_markers("OTHERCIT and OTHERCIT", [("OTHERCIT", "1"), ("OTHERCIT", "2")],
                              {"1": "a", "2": "b"}, own_refid="own") == {}


class _TablePPMI:
    def __init__(self, table):
        self.table = table

    def aggregate(self, c, anchors, how="mean"):
        if not anchors:
            return 0.0
        return sum(self.table.get((c, a), 0.0) for a in anchors) / len(anchors)


def _pool(scores, citing, ctx):
    return {"base": scores, "z": scores, "gold": "g", "anchors": frozenset(),
            "citing_id": citing, "context_id": ctx}


def test_gain_with_scores_only_the_pools_that_have_an_anchor_in_the_set():
    # The proximity table asks "what is an anchor of THIS tier worth", so a
    # pool with no such anchor must not be averaged in as a zero.
    ppmi = _TablePPMI({("g", "near"): 5.0})
    pools = [_pool({"g": 1.0, "x": 2.0}, "p1", "c1"), _pool({"g": 1.0, "x": 2.0}, "p2", "c2")]
    sets = {"c1": {"near"}, "c2": set()}
    got = gain_with(pools, sets, ppmi, lam=1.0, seed=0)
    assert got["n"] == 1 and got["papers"] == 1
    assert got["b"] == pytest.approx(0.5) and got["a"] == pytest.approx(1.0)
    assert got["pool_delta"] == pytest.approx(0.5)
    assert got["mean_anchors"] == 1.0


def test_tiers_are_ordered_nearest_first():
    assert TIERS == ("group", "sentence", "beyond", "not_in_window")
