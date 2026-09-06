import pytest

from locus.experiments.anchor_analysis import (
    CATEGORIES,
    band_cosine,
    band_ppmi,
    categorise,
    grid,
    null_decomposition,
    strata,
    topical_split,
)


def _row(ctx="c1", gold="g", anchor="a", ppmi=0.0, sim=0.9, d_rr=0.0,
         d_score=0.0):
    return {"context_id": ctx, "citing_id": "P", "gold": gold, "anchor": anchor,
            "n_anchors": 2, "ppmi": ppmi, "d_score": d_score, "d_rr": d_rr,
            "d_hit": 0.0, "sim": sim, "set_size": 2, "degree": None}


class TestBanding:
    def test_zero_ppmi_is_its_own_band_not_the_bottom_of_a_continuum(self):
        # ~60% of pairs never co-occur. A plain tercile split would spread the
        # zeros over the bottom two bands and erase the distinction the whole
        # analysis rests on: no evidence versus weak evidence.
        assert band_ppmi(0.0, 9.0) == "zero"
        assert band_ppmi(0.001, 9.0) == "low"
        assert band_ppmi(9.0, 9.0) == "high"

    def test_cosine_bands_are_half_open_and_cover_the_line(self):
        lo, hi = 0.88, 0.95
        assert band_cosine(0.5, lo, hi) == "low"
        assert band_cosine(lo, lo, hi) == "mid"
        assert band_cosine(hi, lo, hi) == "high"
        assert band_cosine(1.0, lo, hi) == "high"


class TestGrid:
    def test_every_pair_with_a_similarity_lands_in_exactly_one_cell(self):
        rows = [_row(ppmi=p, sim=s, ctx=f"c{i}")
                for i, (p, s) in enumerate(
                    [(0.0, 0.1), (0.0, 0.9), (2.0, 0.5), (9.0, 0.2),
                     (9.0, 0.95), (0.0, 0.5), (1.0, 0.99), (5.0, 0.3),
                     (0.0, 0.7)])]
        g = grid(rows)
        assert sum(c["n"] for c in g["cells"].values()) == g["n_with_similarity"]
        assert g["n_with_similarity"] == len(rows)

    def test_pairs_without_a_similarity_are_excluded_not_defaulted(self):
        # 0.04% of pairs have no vector. Defaulting them to 0.0 would drop
        # every one of them into the low-cosine band and inflate exactly the
        # cell the analysis is most tempted to over-claim.
        rows = [_row(sim=None), _row(sim=0.9)]
        g = grid(rows)
        assert g["n_with_similarity"] == 1
        assert sum(c["n"] for c in g["cells"].values()) == 1

    def test_cell_means_are_over_that_cell_only(self):
        rows = [_row(ppmi=0.0, sim=0.99, d_rr=1.0, ctx="a"),
                _row(ppmi=0.0, sim=0.99, d_rr=3.0, ctx="b"),
                _row(ppmi=0.0, sim=0.01, d_rr=-5.0, ctx="c")]
        cells = grid(rows)["cells"]
        assert cells["ppmi_zero|cos_high"]["mean_d_rr"] == pytest.approx(2.0)


class TestNullDecomposition:
    def test_a_zero_ppmi_null_is_sparsity_not_redundancy(self):
        # The distinction the split exists for: nothing was lost because there
        # was nothing to lose, which is a statement about graph coverage, not
        # about the citation set being over-determined.
        rows = [_row(ppmi=0.0, d_rr=0.0, anchor="a1"),
                _row(ppmi=5.0, d_rr=0.3, anchor="a2")]
        c = null_decomposition(rows)["counts"]
        assert c["sparsity"] == 1
        assert c["redundancy_relative"] == 0

    def test_covering_is_directional_the_weak_does_not_cover_the_strong(self):
        # a1 (PPMI 5) is covered by a1's stronger sibling a2 (PPMI 9), but not
        # the reverse: removing the strongest anchor leaves strictly less
        # evidence behind, so calling that redundancy would say the set was
        # over-determined when it has just lost its best evidence.
        rows = [_row(ppmi=5.0, d_rr=0.0, anchor="a1"),
                _row(ppmi=9.0, d_rr=0.0, anchor="a2"),
                _row(ppmi=9.0, d_rr=0.5, anchor="a3", ctx="c2")]
        c = null_decomposition(rows)["counts"]
        assert c["redundancy_relative"] == 1
        assert c["subthreshold_relative"] == 1
        assert c["sparsity"] == 0

    def test_the_two_covering_definitions_can_disagree(self):
        # Two weak anchors cover each other relatively but neither clears the
        # absolute bar. Reporting only `relative` would call this redundancy;
        # only `absolute` would call it sub-threshold. Both are defensible,
        # which is why both are reported.
        rows = [_row(ppmi=1.0, d_rr=0.0, anchor="a1"),
                _row(ppmi=1.0, d_rr=0.0, anchor="a2")]
        rows += [_row(ppmi=50.0, d_rr=0.9, anchor=f"b{i}", ctx=f"x{i}")
                 for i in range(20)]
        c = null_decomposition(rows)["counts"]
        assert c["redundancy_relative"] == 2
        assert c["redundancy_absolute"] == 0
        assert c["subthreshold_absolute"] == 2

    def test_an_anchor_does_not_cover_itself(self):
        # The only anchor with evidence, removed, leaves nothing behind. If the
        # sibling scan included the removed row it would report redundancy for
        # a set that has just lost all of its evidence.
        rows = [_row(ppmi=7.0, d_rr=0.0, anchor="a1"),
                _row(ppmi=0.0, d_rr=0.0, anchor="a2")]
        c = null_decomposition(rows)["counts"]
        assert c["redundancy_relative"] == 0
        assert c["subthreshold_relative"] == 1
        assert c["sparsity"] == 1

    def test_non_null_interventions_are_not_counted(self):
        rows = [_row(ppmi=3.0, d_rr=0.4), _row(ppmi=0.0, d_rr=-0.2, anchor="b")]
        n = null_decomposition(rows)
        assert n["n_null"] == 0
        assert sum(n["counts"].values()) == 0


class TestCategorise:
    def test_the_three_categories_partition_every_intervention(self):
        # A taxonomy that leaves rows unclassified is not a partition, and the
        # earlier five-way version silently dropped ~9% of interventions into
        # none of its categories.
        rows = [_row(ppmi=5.0, d_rr=0.4, anchor="a"),
                _row(ppmi=5.0, d_rr=0.0, anchor="b"),
                _row(ppmi=5.0, d_rr=-0.2, anchor="c"),
                _row(ppmi=0.0, d_rr=0.0, anchor="d"),
                _row(ppmi=0.0, d_rr=-0.3, anchor="e")]
        cats = categorise(rows)
        assert sum(len(v) for v in cats.values()) == len(rows)
        assert set(cats) == set(CATEGORIES)

    def test_a_costless_removal_with_evidence_is_redundant_not_informative(self):
        rows = [_row(ppmi=9.0, d_rr=0.0), _row(ppmi=9.0, d_rr=-0.1, anchor="b")]
        cats = categorise(rows)
        assert len(cats["redundant"]) == 2
        assert cats["informative"] == []

    def test_zero_ppmi_is_non_cocited_whatever_the_rank_did(self):
        # Category membership is about the evidence available, not the outcome:
        # a zero-PPMI anchor that happens to coincide with a rank change is
        # still an anchor the graph knows nothing about.
        rows = [_row(ppmi=0.0, d_rr=0.5), _row(ppmi=0.0, d_rr=-0.5, anchor="b")]
        assert len(categorise(rows)["non_cocited"]) == 2


class TestTopicalSplit:
    def _peaked(self, centres, spread=0.02, n=40):
        """Rows whose cosines form a triangular bump around each centre.

        `n` is deliberately well above the 20 histogram bins: with one
        distinct value per bin the counts develop gaps that read as extra
        modes, which is a property of the fixture rather than of the shape.
        """
        rows, k = [], 0
        for c in centres:
            for d in range(-n, n + 1):
                for _ in range(n + 1 - abs(d)):
                    rows.append(_row(ppmi=9.0, d_rr=0.4, anchor=f"a{k}",
                                     ctx=f"c{k}", sim=c + d * spread / n))
                    k += 1
        return rows

    def test_a_unimodal_input_reports_one_mode(self):
        # The claim the sub-split rests on: informative anchors are a
        # continuum in cosine, so reporting them as two types would be
        # inventing a mode the data does not have.
        assert topical_split(self._peaked([0.93]), 0.93)["modes"] == 1

    def test_a_genuinely_bimodal_input_reports_two(self):
        # Without this the modality check could be a constant returning 1, and
        # "the distribution is unimodal" would be a claim about the code.
        assert topical_split(self._peaked([0.88, 0.96]), 0.92)["modes"] == 2

    def test_modes_at_the_edges_of_the_range_are_counted(self):
        # An earlier detector scanned interior bins only and reported ZERO
        # modes for a distribution whose peaks both sat near the ends -- the
        # most confidently wrong answer available.
        from locus.experiments.anchor_analysis import _count_modes
        assert _count_modes([20, 12, 3, 1, 1, 3, 12, 20]) == 2

    def test_a_one_count_wobble_in_the_tail_is_not_a_mode(self):
        from locus.experiments.anchor_analysis import _count_modes
        assert _count_modes([1, 0, 1, 0, 40, 90, 40, 0]) == 1

    def test_rows_without_similarity_are_excluded_from_the_split(self):
        rows = [_row(ppmi=9.0, d_rr=0.4, sim=None),
                _row(ppmi=9.0, d_rr=0.4, sim=0.95, anchor="b")]
        assert topical_split(rows, 0.9)["n_known_similarity"] == 1


class TestStrata:
    def _rows(self):
        rows = [   # non-co-cited, harmful
            _row(ctx=f"c{i}", anchor=f"a{i}", ppmi=0.0,
                 sim=0.90 + i / 1000, d_rr=-0.1)
            for i in range(60)
        ]
        rows.extend(   # informative, cross-topic
            _row(ctx=f"d{i}", anchor=f"b{i}", ppmi=10.0,
                 sim=0.10 + i / 1000, d_rr=0.9)
            for i in range(60)
        )
        return rows

    def test_sampling_is_deterministic_for_a_fixed_seed(self):
        rows = self._rows()
        a = strata(rows, 0.5, seed=0)
        b = strata(rows, 0.5, seed=0)
        for k in a:
            if k.startswith("_"):
                continue
            assert [r["anchor"] for r in a[k]["sample"]] == \
                [r["anchor"] for r in b[k]["sample"]]

    def test_a_different_seed_draws_a_different_sample(self):
        # Otherwise "fixed seed" is decoration and the sample is not actually
        # a draw from the stratum.
        rows = self._rows()
        a = strata(rows, 0.5, seed=0)["informative_cross_topic"]["sample"]
        b = strata(rows, 0.5, seed=7)["informative_cross_topic"]["sample"]
        assert [r["anchor"] for r in a] != [r["anchor"] for r in b]

    def test_population_is_reported_alongside_the_sample(self):
        # The anti-cherry-pick guarantee is the population count, not the
        # sample: 3 examples drawn from 12 and from 12,000 read identically
        # on the page and mean entirely different things.
        rows = self._rows()
        st = strata(rows, 0.5, seed=0, per_stratum=3)
        assert st["informative_cross_topic"]["population"] == 60
        assert len(st["informative_cross_topic"]["sample"]) == 3

    def test_sampling_never_exceeds_a_thin_stratum(self):
        rows = [_row(ppmi=0.0, sim=0.99, d_rr=-0.1)]
        st = strata(rows, 0.5, seed=0, per_stratum=10)
        assert len(st["non_cocited_harmful"]["sample"]) == 1
        assert st["informative_cross_topic"]["sample"] == []

    def test_rows_without_a_similarity_are_excluded_and_counted(self):
        # 64 real interventions have no vector. An example printed with a
        # blank cosine is not an example of anything on the cosine axis.
        rows = [_row(ppmi=9.0, d_rr=0.5, sim=None),
                _row(ppmi=9.0, d_rr=0.5, sim=0.99, anchor="b")]
        st = strata(rows, 0.5, seed=0)
        assert st["_excluded_no_similarity"] == 1
        assert st["informative_aligned"]["population"] == 1


class TestAnchorMap:
    """Figure 4: mean rank influence on a fixed PPMI-band x cosine-octile grid.

    The paper's grid was produced once and transcribed into figure_4.tex; this
    is the code that regenerates it, so its conventions are pinned here: PPMI
    edges are fixed (zero is its own band, then 7.5, 8.5, 9, 9.5, 10, 11), the
    cosine edges are the eight quantiles of the observed similarities, and the
    top band and the rightmost octile are closed so the maximum lands in a cell.
    """

    def test_ppmi_edges_are_fixed_and_the_top_edge_clears_the_maximum(self):
        from locus.experiments.anchor_analysis import PPMI_EDGES, anchor_map
        rows = [_row(ppmi=p, sim=0.9, ctx=f"c{i}")
                for i, p in enumerate([0.0, 3.0, 8.0, 12.363])]
        m = anchor_map(rows)
        assert m["ye"][:-1] == list(PPMI_EDGES)
        assert PPMI_EDGES == (0.0, 1e-9, 7.5, 8.5, 9.0, 9.5, 10.0, 11.0)
        assert m["ye"][-1] == pytest.approx(13.363)

    def test_cosine_edges_are_the_octiles_of_the_observed_similarities(self):
        from locus.experiments.anchor_analysis import anchor_map
        sims = [i / 16 for i in range(17)]          # 0, 1/16, ..., 1
        rows = [_row(sim=s, ctx=f"c{i}") for i, s in enumerate(sims)]
        m = anchor_map(rows)
        assert len(m["xe"]) == 9
        assert m["xe"][0] == 0.0 and m["xe"][-1] == 1.0
        assert m["xe"][4] == pytest.approx(0.5)

    def test_every_row_with_a_similarity_lands_in_exactly_one_cell(self):
        from locus.experiments.anchor_analysis import anchor_map
        rows = [_row(ppmi=p, sim=s, ctx=f"c{i}") for i, (p, s) in enumerate(
            [(0.0, 0.1), (0.0, 1.0), (12.0, 1.0), (12.0, 0.1), (7.5, 0.5),
             (9.0, 0.9), (0.5, 0.3), (11.0, 0.99)])]
        rows.append(_row(sim=None, ppmi=9.0))     # no vector: excluded
        m = anchor_map(rows)
        assert sum(n for _lo, _hi, cells in m["grid"] for n, _mean in cells) == 8

    def test_cell_mean_is_over_that_cell_only_and_bands_run_top_down(self):
        from locus.experiments.anchor_analysis import anchor_map
        rows = [_row(ppmi=12.0, sim=0.99, d_rr=0.5, ctx="a"),
                _row(ppmi=12.0, sim=0.99, d_rr=0.1, ctx="b"),
                _row(ppmi=0.0, sim=0.01, d_rr=-1.0, ctx="c"),
                _row(ppmi=8.0, sim=0.5, d_rr=0.0, ctx="d")]
        m = anchor_map(rows)
        top_lo, _top_hi, top_cells = m["grid"][0]
        assert top_lo == 11.0                       # first row is the >= 11 band
        assert top_cells[-1] == (2, pytest.approx(0.3))
        bottom_lo, _bottom_hi, bottom_cells = m["grid"][-1]
        assert bottom_lo == 0.0
        assert bottom_cells[0] == (1, -1.0)
        assert all(cell == (0, None) for cell in top_cells[:-1])

    def test_pgfplots_table_matches_figure_4_layout(self):
        # figure_4.tex reads `x y value` triples, x = cosine octile 0..7,
        # y = PPMI band 0..7 with 0 the zero band and 7 the >= 11 band, one
        # blank line between bands, values at 4 dp. Empty cells print `nan`.
        from locus.experiments.anchor_analysis import anchor_map, pgfplots_table
        rows = [_row(ppmi=12.0, sim=0.99, d_rr=0.27234, ctx="a"),
                _row(ppmi=0.0, sim=0.01, d_rr=-0.02631, ctx="c")]
        text = pgfplots_table(anchor_map(rows))
        lines = text.splitlines()
        assert lines[0] == "0 7 nan"
        assert lines[7] == "7 7 0.2723"
        assert lines[8] == ""
        assert lines[-1] == "7 0 nan"
        assert "0 0 -0.0263" in lines
