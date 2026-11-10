# Run reports

The JSON files every number in the paper is read from, written by the
commands in [docs/reproduce.md](../docs/reproduce.md) against the frozen
test/val artefacts (seed 0). They carry counts, metrics and intervals only:
no identifiers, no text. Paths inside them have been reduced to file names.

| Paper | File | Keys |
|---|---|---|
| Table 1, SPECTER2 row; Table 10 hard and PPMI rows | `rank_report.json` | `gate1_primary`, `gate1_hit_at_1`, `papers` |
| Table 1, other rows | `rank_report_{scincl,hatten,bm25}.json` | `gate1_primary`, `gate1_hit_at_1`, `selected` (α, min_count, λ) |
| Table 2 | `swap_report_{specter2,scincl,hatten,bm25}.json` | `sigma`, `selected_lambda`, `test.{base,cocite,both,both-base}` |
| Table 3, settings A to D | `expb_report_{revealed,selfseed,revealed_corpuswide,corpuswide}.json` | `arms.*.20`, `deltas.hybrid_retrieved.20` |
| Table 4, oracle-prefetch and corpus-wide rows | `expb_report_{revealed,revealed_corpuswide}.json` | `remaining`, `remaining_deltas`, `remaining_gold_reachability`, `remaining_by_set_size` |
| Table 5 and Table 6; Appendix A threshold paragraphs | `theta_sensitivity.json` | `cohort_contexts`, `fixed_pool_contexts`, `fixed_pool_contexts_with_identical_anchors`, `swap_cohort_contexts`, `thetas.*.{composition,selected,full_rebuild,fixed_cohort,fixed_pools,swapped_anchors,swapped_anchors_changed}` |
| Table 7, by inference route; Section 3.2 | `rank_report_route_{marker,clustering}.json` | `anchor_route`, `pools_with_anchors`, `gate1_primary`, `gate1_on_anchored` |
| Table 7, by anchor proximity; Appendix A | `proximity_specter2.json` | `headline_pools_by_closest_tier`, `gain_one_tier_only`, `gain_fixed_cohort_with_group_anchor`, `marker_route_instances` |
| Table 8; Sections 3.2, 3.3 and 7; Appendix A marker resolution | `build_report_{test,val}.json` | `contexts`, `citing_papers`, `num_locations`, `anchor_ge1_{clustering,marker,union}`, `anchor_distribution`, `alignment_success`, `marker_anchors_{total,unresolvable,implausible,self}`, `rank_items`, `swap_pair_count`, `hard`, `headline`, `null_controls` |
| Table 9 | `rank_report.json` | `coverage_strata.{target,distractor,none,unseen}` |
| Appendix B graph-coverage paragraph, pool-level means | `rank_report.json` | `gate1_pool_level` (with `ci`), `coverage_strata.*.{pool_a,pool_b,pool_delta,pool_ci}` |
| Table 10, easy row | `rank_report_easy.json` | `gate1_primary` |
| Table 10, other location | `rank_report.json` | `other_location_vs_base`, `gate2_other_location_control` |
| Table 10, random / degree-matched | `controls_specter2.json` | `arms.*.{mrr,vs_base}` |
| Table 10, raw co-count / node2vec / sibling titles | `rank_report_{count,node2vec,siblings}.json` | `gate1_primary` |
| Appendix B, hard-versus-easy difference and ratio with joint CIs | `hard_easy_contrast_specter2.json` | `joint.difference`, `joint.ratio`, `joint.within_paper` |
| Appendix B, node2vec Swap grid | `swap_report_node2vec.json` | `val_grid` |
| Appendix B, node2vec in sequential completion | `expb_report_n2v_{revealed,selfseed,corpuswide}_none.json` | `deltas.hybrid_retrieved.20` |
| Section 4.1 graph size and leakage | `graph_report.json` | `papers_counted`, `directed_nonzeros`, `held_out_papers`, `leakage_overlap` |
| Section 5.3 first-stage recall | `retrieve_report_test.json` | `first_stage_recall`, `gold_total` |
| Section 6.1 first-pick hit rate and Recall@20 by seed correctness | `expb_report_selfseed.json`, `expb_report_corpuswide.json` | `seed_quality`, `locations`, `arms.semantic.1` |
| Section 6.2 categories | `anchor_analysis_specter2.json` | `categories`, `topical_split`, `nulls` |
| Figure 3 | `anchor_map_specter2.json`, `anchor_map_specter2.pgf.txt` | `xe`, `ye`, `grid` |
| Appendix C, anchor influence by \|A\| | `influence_specter2.json` | `by_n_anchors` |
| λ / α selection, one sweep per base or association measure | `sweep_val*.json` | `best`, `grid` |
| Annotation sample (docs/annotation_protocol.md) | `annotation_manifest.json` | `cells.*.{population,quota,drawn}` |
| Value-to-report map, checked by `tests/test_paper_map.py` | `paper_map.json` | `claims[].{report,key,paper,expect}` |

Every rank report records `averaging: "paper"`: each metric is a mean of
per-citing-paper means, and intervals resample citing papers. Where a
pool-level mean is also given it is named as such, and its interval
(`pool_ci`) still resamples citing papers, taking the mean over every pool
the resampled papers contribute.

Not here: Figure 2 and the coherence numbers in Sections 1.3 and 5.1 (the
mean pairwise cosine of retrieved against human citation sets), which were
produced outside the harness and have no report.

Only runs that back a number in the paper are kept: the aggregator ablations
(`--agg masked_mean` / `max`) and the node2vec `(p, q)` grid other than the
selected `sweep_val_node2vec.json` are omitted; `docs/reproduce.md` lists how
to rerun them.

`expb_report_*.json` were produced with `--standardize none` (the raw-cosine
base the paper's sequential-completion numbers use; the `zscore` default is
Eq. 1's scale and gives slightly larger gains) and 1000 bootstrap resamples.
