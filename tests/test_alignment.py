from locus.core.alignment import (
    MAX_PLAUSIBLE_REF,
    build_bibliography,
    markers_with_numbers,
    resolve,
)


class TestMarkersWithNumbers:
    def test_single_target(self):
        raw = "as shown in [4] the effect holds"
        masked = "as shown in TARGETCIT the effect holds"
        assert markers_with_numbers(raw, masked) == [("TARGETCIT", "4")]

    def test_adjacent_run_preserves_order(self):
        raw = "parallaxes) [4] [5] and Gaia DR2"
        masked = "parallaxes) TARGETCIT OTHERCIT and Gaia DR2"
        assert markers_with_numbers(raw, masked) == [
            ("TARGETCIT", "4"),
            ("OTHERCIT", "5"),
        ]

    def test_target_slot_moved_in_sibling_context(self):
        raw = "parallaxes) [4] [5] and Gaia DR2"
        masked = "parallaxes) OTHERCIT TARGETCIT and Gaia DR2"
        assert markers_with_numbers(raw, masked) == [
            ("OTHERCIT", "4"),
            ("TARGETCIT", "5"),
        ]

    def test_marker_at_end_of_window(self):
        raw = "prior work [12]"
        masked = "prior work TARGETCIT"
        assert markers_with_numbers(raw, masked) == [("TARGETCIT", "12")]

    def test_count_mismatch_returns_none(self):
        # two markers but only one number between the anchors
        raw = "see [7] here"
        masked = "see TARGETCIT OTHERCIT here"
        assert markers_with_numbers(raw, masked) is None

    def test_unfindable_segment_returns_none(self):
        assert markers_with_numbers("totally different", "x TARGETCIT y") is None


class TestBuildBibliography:
    def test_maps_target_numbers_to_refids(self):
        bib = build_bibliography(
            [
                ("A", [("TARGETCIT", "1")]),
                ("B", [("TARGETCIT", "2"), ("OTHERCIT", "1")]),
            ]
        )
        assert bib.number_to_ref == {"1": "A", "2": "B"}
        assert bib.conflicted_numbers == frozenset()
        assert bib.n_entries == 2

    def test_one_number_two_refids_is_a_conflict(self):
        bib = build_bibliography(
            [("A", [("TARGETCIT", "1")]), ("B", [("TARGETCIT", "1")])]
        )
        assert "1" in bib.conflicted_numbers
        assert "1" not in bib.number_to_ref

    def test_one_refid_two_numbers_is_a_conflict(self):
        bib = build_bibliography(
            [("A", [("TARGETCIT", "1")]), ("A", [("TARGETCIT", "7")])]
        )
        assert bib.conflicted_numbers == frozenset({"1", "7"})
        assert bib.number_to_ref == {}

    def test_repeating_the_same_pair_is_not_a_conflict(self):
        bib = build_bibliography(
            [("A", [("TARGETCIT", "1")]), ("A", [("TARGETCIT", "1")])]
        )
        assert bib.number_to_ref == {"1": "A"}
        assert bib.conflicted_numbers == frozenset()

    def test_contexts_without_exactly_one_target_are_skipped(self):
        bib = build_bibliography(
            [
                ("A", [("OTHERCIT", "1")]),
                ("B", [("TARGETCIT", "2"), ("TARGETCIT", "3")]),
                ("C", [("TARGETCIT", "4")]),
            ]
        )
        assert bib.number_to_ref == {"4": "C"}

    def test_empty(self):
        bib = build_bibliography([])
        assert bib.number_to_ref == {} and bib.n_entries == 0


class TestResolve:
    def _bib(self):
        return build_bibliography(
            [
                ("A", [("TARGETCIT", "1")]),
                ("B", [("TARGETCIT", "2")]),
                ("C", [("TARGETCIT", "3")]),
            ]
        )

    def test_valid_target_and_resolved_anchors(self):
        r = resolve([("TARGETCIT", "1"), ("OTHERCIT", "2")], self._bib(), "A")
        assert r.target_valid
        assert r.anchors == frozenset({"B"})
        assert r.n_unresolvable == 0
        assert r.n_implausible == 0

    def test_target_number_disagreeing_with_refid_is_invalid(self):
        # alignment scraped "2" but the dataset says this context targets A
        r = resolve([("TARGETCIT", "2")], self._bib(), "A")
        assert not r.target_valid

    def test_target_on_a_conflicted_number_is_invalid(self):
        bib = build_bibliography(
            [("A", [("TARGETCIT", "1")]), ("B", [("TARGETCIT", "1")])]
        )
        assert not resolve([("TARGETCIT", "1")], bib, "A").target_valid

    def test_unknown_anchor_number_is_unresolvable_not_an_error(self):
        r = resolve([("TARGETCIT", "1"), ("OTHERCIT", "99")], self._bib(), "A")
        assert r.target_valid
        assert r.anchors == frozenset()
        assert r.n_unresolvable == 1

    def test_implausible_number_is_counted_separately(self):
        # a year scraped out of prose, not a reference number
        r = resolve([("TARGETCIT", "1"), ("OTHERCIT", "2019")], self._bib(), "A")
        assert r.n_implausible == 1
        assert r.n_unresolvable == 0

    def test_anchor_equal_to_own_target_is_dropped(self):
        r = resolve([("TARGETCIT", "1"), ("OTHERCIT", "1")], self._bib(), "A")
        assert r.anchors == frozenset()

    def test_missing_target_marker_is_invalid(self):
        assert not resolve([("OTHERCIT", "2")], self._bib(), "A").target_valid


class TestImplausibleTargetNumbers:
    def test_a_year_never_becomes_a_bibliography_entry(self):
        # resolve() already rejects an OTHERCIT above MAX_PLAUSIBLE_REF as a
        # scraped year. build_bibliography must apply the same filter, or the
        # same number is simultaneously a trusted bibliography key and an
        # implausible anchor.
        entries = [
            ("R1", [("TARGETCIT", "2019")]),
            ("R1", [("TARGETCIT", "2019")]),
        ]
        bib = build_bibliography(entries)
        assert "2019" not in bib.number_to_ref

    def test_a_plausible_number_still_maps(self):
        bib = build_bibliography([("R1", [("TARGETCIT", "16")])])
        assert bib.number_to_ref == {"16": "R1"}

    def test_the_boundary_is_inclusive(self):
        n = str(MAX_PLAUSIBLE_REF)
        assert build_bibliography([("R1", [("TARGETCIT", n)])]).number_to_ref == {n: "R1"}
        over = str(MAX_PLAUSIBLE_REF + 1)
        assert build_bibliography([("R1", [("TARGETCIT", over)])]).number_to_ref == {}

    def test_a_context_keyed_on_a_year_does_not_validate(self):
        entries = [("R1", [("TARGETCIT", "2019")])] * 2
        bib = build_bibliography(entries)
        res = resolve([("TARGETCIT", "2019")], bib, "R1")
        assert res.target_valid is False


class TestSelfAnchorAccounting:
    def test_an_othercit_pointing_at_the_target_is_counted(self):
        # Currently such a marker lands in neither anchors nor either failure
        # counter, so the build report's anchor accounting does not sum.
        bib = build_bibliography([("R1", [("TARGETCIT", "7")])])
        res = resolve([("TARGETCIT", "7"), ("OTHERCIT", "7")], bib, "R1")
        assert res.anchors == frozenset()
        assert res.n_self == 1

    def test_the_buckets_partition_every_othercit(self):
        bib = build_bibliography(
            [("R1", [("TARGETCIT", "7")]), ("R2", [("TARGETCIT", "8")])]
        )
        markers = [
            ("TARGETCIT", "7"),
            ("OTHERCIT", "8"),      # resolves to another paper -> anchor
            ("OTHERCIT", "7"),      # resolves to self          -> n_self
            ("OTHERCIT", "99"),     # not in the bibliography   -> unresolvable
            ("OTHERCIT", "2019"),   # above MAX_PLAUSIBLE_REF   -> implausible
        ]
        res = resolve(markers, bib, "R1")
        n_othercit = sum(1 for t, _ in markers if t == "OTHERCIT")
        assert (
            len(res.anchors) + res.n_self + res.n_unresolvable + res.n_implausible
            == n_othercit
        )
