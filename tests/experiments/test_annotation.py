"""The annotation sample: which pairs two readers label, and what they see.

Locations and anchors are inferred, and the only way to know how often the
inference is right is to ask a reader. The sample has to cover the pipeline's
likely MERGES and its likely SPLITS, so it is stratified by what the pipeline
decided and why, and each item must be presentable without telling the reader
what that decision was.
"""
import collections
import csv

import pytest

from locus.core.types import ContextRec
from locus.experiments.annotation import (
    QUOTAS,
    STRATA,
    draw,
    order_for,
    pairs_in_paper,
    render,
    stratum_of,
    write_sheet,
)

# ---- which stratum a pair belongs to ----------------------------------------


def test_strata_are_exhaustive_and_the_quotas_sum_to_four_hundred():
    assert set(QUOTAS) == set(STRATA)
    assert sum(QUOTAS.values()) == 400


@pytest.mark.parametrize(
    ("kw", "expected"),
    [
        (dict(clustering=True, marker=True, direct=True, aligned=True, jaccard=0.9), "both_routes"),
        (dict(clustering=False, marker=True, direct=False, aligned=True, jaccard=0.3), "marker_only"),
        (dict(clustering=True, marker=False, direct=True, aligned=True, jaccard=0.7),
         "clustering_only_aligned"),
        (dict(clustering=True, marker=False, direct=True, aligned=False, jaccard=0.7),
         "clustering_only_unaligned"),
        (dict(clustering=False, marker=False, direct=False, aligned=True, jaccard=0.35), "near_miss"),
        (dict(clustering=False, marker=False, direct=False, aligned=True, jaccard=0.1),
         "weak_overlap"),
        (dict(clustering=False, marker=False, direct=False, aligned=True, jaccard=0.0), "disjoint"),
    ],
)
def test_stratum_of(kw, expected):
    assert stratum_of(**kw) == expected


def test_a_transitive_join_outranks_the_route_strata():
    # In one cluster only because a third window links them: the pairwise
    # similarity is below the threshold. This is where union-find can
    # over-merge, so it is its own stratum whichever routes agree.
    for marker in (True, False):
        assert stratum_of(clustering=True, marker=marker, direct=False, aligned=True,
                          jaccard=0.2) == "transitive"


# ---- what the reader sees ---------------------------------------------------


def test_render_marks_the_target_and_the_candidate_and_hides_the_rest():
    masked = "as shown in OTHERCIT and  TARGETCIT  , unlike OTHERCIT [19] ."
    marks = [("OTHERCIT", "3"), ("TARGETCIT", "4"), ("OTHERCIT", "7")]
    got = render(masked, marks, {"3": "pA", "4": "pT", "7": "pB"}, candidate="pA")
    assert got == "as shown in ⟦C⟧ and  ⟦T⟧  , unlike [cit] [19] ."


def test_render_marks_every_occurrence_of_the_candidate():
    masked = "OTHERCIT then TARGETCIT then OTHERCIT"
    marks = [("OTHERCIT", "3"), ("TARGETCIT", "4"), ("OTHERCIT", "3")]
    assert render(masked, marks, {"3": "pA", "4": "pT"}, candidate="pA") == \
        "⟦C⟧ then ⟦T⟧ then ⟦C⟧"


def test_render_without_alignment_still_shows_the_target():
    # No marker numbers were recovered, so the candidate cannot be located in
    # this window; the reader gets it from the candidate's own window instead.
    assert render("x OTHERCIT y TARGETCIT", None, {}, candidate="pA") == "x [cit] y ⟦T⟧"


def test_render_can_swap_the_roles_for_the_candidates_own_window():
    # The candidate's own window has the candidate in the target slot.
    masked = "see OTHERCIT , TARGETCIT"
    marks = [("OTHERCIT", "4"), ("TARGETCIT", "3")]
    got = render(masked, marks, {"3": "pA", "4": "pT"}, candidate="pT",
                 own="⟦C⟧", other="⟦T⟧")
    assert got == "see ⟦T⟧ , ⟦C⟧"


# ---- classifying the pairs of one citing paper ------------------------------


def _window(lo: int, hi: int, slots: dict[int, list[tuple[str, str]]]) -> tuple[str, str]:
    """(raw, masked) for words lo..hi with markers after the given positions.

    slots: word index -> [(kind, number), ...]. Shingles keep letters only,
    so each word is spelled from its index in letters.
    """
    raw, masked = [], []
    for i in range(lo, hi):
        word = "".join(chr(ord("a") + int(d)) for d in f"{i:02d}") + "x"
        raw.append(word)
        masked.append(word)
        for kind, num in slots.get(i, ()):
            raw.append(f"[{num}]")
            masked.append(kind)
    return " ".join(raw), " ".join(masked)


def _rec(cid, refid, location, lo, hi, slots, **kw):
    raw, masked = _window(lo, hi, slots)
    return ContextRec(context_id=cid, citing_id="P", refid=refid, raw=raw, masked=masked,
                      location=location, **kw)


def _paper():
    # c1 and c2: one bracket, [1] and [2], nearly the same window -> both routes.
    # c3: cites [3] in a window overlapping c1's by about a third -> near miss.
    # c4: cites [4] somewhere else entirely -> disjoint.
    c1 = _rec("c1", "p1", 0, 0, 30, {14: [("TARGETCIT", "1"), ("OTHERCIT", "2")]},
              marker_anchors=frozenset({"p2"}))
    c2 = _rec("c2", "p2", 0, 0, 30, {14: [("OTHERCIT", "1"), ("TARGETCIT", "2")]},
              marker_anchors=frozenset({"p1"}))
    c3 = _rec("c3", "p3", 1, 12, 42, {30: [("TARGETCIT", "3")]})
    c4 = _rec("c4", "p4", 2, 50, 80, {60: [("TARGETCIT", "4")]})
    return [c1, c2, c3, c4]


def test_pairs_in_paper_classifies_every_ordered_pair():
    rows = pairs_in_paper(_paper())
    by = {(r["context_id"], r["candidate"]): r for r in rows}
    assert len(rows) == 12                       # 4 contexts x 3 other cited papers

    both = by[("c1", "p2")]
    assert both["stratum"] == "both_routes"
    assert both["tier"] == "group"               # nothing but a space between the markers
    assert both["candidate_context"] == "c2"
    assert both["jaccard"] > 0.5

    near = by[("c1", "p3")]
    assert near["stratum"] == "near_miss"
    assert 0.2 <= near["jaccard"] < 0.5
    assert near["tier"] == "not_in_window"

    assert by[("c1", "p4")]["stratum"] == "disjoint"
    assert by[("c1", "p4")]["jaccard"] == 0.0


def test_pairs_in_paper_carries_the_text_the_reader_needs():
    by = {(r["context_id"], r["candidate"]): r for r in pairs_in_paper(_paper())}
    both = by[("c1", "p2")]
    assert "⟦T⟧ ⟦C⟧" in both["text"]
    assert both["text_2"] == ""                  # the candidate is visible in the window
    near = by[("c1", "p3")]
    assert "⟦T⟧" in near["text"] and "⟦C⟧" not in near["text"]
    assert "⟦C⟧" in near["text_2"]               # shown in its own window instead


def test_a_paper_citing_one_paper_twice_yields_one_pair_per_candidate():
    recs = _paper()
    again = _rec("c5", "p2", 2, 50, 80, {61: [("TARGETCIT", "2")]})
    by = collections.Counter(
        (r["context_id"], r["candidate"]) for r in pairs_in_paper([*recs, again]))
    assert by[("c1", "p2")] == 1
    # ...and the window shown for the candidate is the one nearest the target.
    row = next(r for r in pairs_in_paper([*recs, again])
               if (r["context_id"], r["candidate"]) == ("c1", "p2"))
    assert row["candidate_context"] == "c2"


# ---- drawing the sample -----------------------------------------------------


def _population(n_papers=40, per_paper=6):
    return [{"context_id": f"p{p}c{i}", "citing_id": f"p{p}", "candidate": f"r{i}",
             "stratum": "near_miss", "tier": "not_in_window"}
            for p in range(n_papers) for i in range(per_paper)]


def test_draw_is_deterministic_and_fills_the_quota():
    cells = {("near_miss", "all"): _population()}
    a = draw(cells, {("near_miss", "all"): 30}, seed=7)
    b = draw(cells, {("near_miss", "all"): 30}, seed=7)
    assert a == b and len(a["items"]) == 30
    assert draw(cells, {("near_miss", "all"): 30}, seed=8)["items"] != a["items"]


def test_draw_takes_at_most_two_items_from_one_citing_paper():
    cells = {("near_miss", "all"): _population(), ("disjoint", "all"): _population()}
    got = draw(cells, {("near_miss", "all"): 30, ("disjoint", "all"): 30}, seed=0)
    per_paper = collections.Counter(it["citing_id"] for it in got["items"])
    assert max(per_paper.values()) <= 2


def test_draw_never_reuses_an_excluded_pair():
    cells = {("near_miss", "all"): _population()}
    pilot = draw(cells, {("near_miss", "all"): 20}, seed=1)
    used = {(it["context_id"], it["candidate"]) for it in pilot["items"]}
    main = draw(cells, {("near_miss", "all"): 30}, seed=0, exclude=used)
    assert not used & {(it["context_id"], it["candidate"]) for it in main["items"]}


def test_draw_reports_a_cell_it_could_not_fill_rather_than_padding_it():
    cells = {("near_miss", "all"): _population(n_papers=3)}      # 6 items at most
    got = draw(cells, {("near_miss", "all"): 30}, seed=0)
    assert len(got["items"]) == 6
    assert got["cells"][("near_miss", "all")] == {"population": 18, "quota": 30, "drawn": 6}


def test_draw_records_the_population_each_item_stands_for():
    cells = {("near_miss", "all"): _population()}
    got = draw(cells, {("near_miss", "all"): 30}, seed=0)
    assert got["cells"][("near_miss", "all")]["population"] == 240
    assert all(it["cell"] == "near_miss/all" for it in got["items"])


# ---- the sheets -------------------------------------------------------------


def test_each_annotator_gets_the_same_items_in_a_different_order():
    ids = [f"L{i:03d}" for i in range(50)]
    a, b = order_for(ids, "A", seed=0), order_for(ids, "B", seed=0)
    assert sorted(a) == sorted(b) == ids
    assert a != b
    assert a == order_for(ids, "A", seed=0)


def test_the_sheet_shows_the_text_and_nothing_about_the_pipeline(tmp_path):
    items = [{"item_id": "L001", "text": "x ⟦T⟧ ⟦C⟧", "text_2": "",
              "stratum": "both_routes", "tier": "group", "jaccard": 0.9,
              "context_id": "c1", "candidate": "p2", "citing_id": "P"}]
    path = tmp_path / "sheet.csv"
    write_sheet(path, items, ["L001"])
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["item_id", "text", "text_2", "label", "uncertain", "comment"]
    assert rows[0]["text"] == "x ⟦T⟧ ⟦C⟧" and rows[0]["label"] == ""


# ---- how a stratum's quota is spread over its cells -------------------------


def test_cell_quotas_keep_each_stratum_total():
    from locus.experiments.annotation import PILOT_QUOTAS, cell_quotas

    for quotas in (QUOTAS, PILOT_QUOTAS):
        per_stratum = collections.Counter()
        for (stratum, _tier), n in cell_quotas(quotas).items():
            per_stratum[stratum] += n
        assert dict(per_stratum) == quotas
    assert sum(PILOT_QUOTAS.values()) == 40


def test_agreeing_routes_are_balanced_over_the_proximity_tiers():
    from locus.experiments.annotation import cell_quotas

    got = cell_quotas(QUOTAS)
    assert got["both_routes", "group"] == got["both_routes", "sentence"] \
        == got["both_routes", "beyond"] == 20


def test_marker_only_pairs_in_one_group_get_a_small_share():
    # A marker in the target's own citation group but a different cluster is
    # an anomaly: a few hundred pairs in the whole split. It is sampled, so
    # the estimates cover it, but an even three-way split would spend a third
    # of the stratum on it.
    from locus.experiments.annotation import cell_quotas

    got = cell_quotas(QUOTAS)
    assert got["marker_only", "group"] == 10
    assert got["marker_only", "sentence"] == got["marker_only", "beyond"] == 35
