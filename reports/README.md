# Run reports

The JSON files every number in the paper is read from, written by the
commands in [docs/reproduce.md](../docs/reproduce.md) against the frozen
test/val artefacts (seed 0). They carry counts, metrics and intervals only:
no identifiers, no text. Paths inside them have been reduced to file names.

| Paper | File | Keys |
|---|---|---|
| Table 1, SPECTER2 row; Table 7 hard row | `rank_report.json` | `gate1_primary`, `gate1_hit_at_1` |
| Table 1, other rows | `rank_report_{scincl,hatten,bm25}.json` | `gate1_primary`, `gate1_hit_at_1`, `selected` (α, min_count, λ) |
| Table 2 | `swap_report_{specter2,scincl,hatten,bm25}.json` | `sigma`, `selected_lambda`, `test.{base,cocite,both,both-base}` |
| Table 3 | `expb_report_{revealed,selfseed,corpuswide}.json` | `arms.*.20`, `deltas.hybrid_retrieved.20` |
| Table 4 (N per stratum) | `anchor_analysis_specter2.json` | `strata.*.population`, `categories` |
| Table 5 | `build_report_{test,val}.json` | `contexts`, `citing_papers`, `num_locations`, `anchor_ge1_union`, `rank_items`, `swap_pair_count`, `hard`, `headline`, `null_controls` |
| Table 6 | `rank_report.json` | `coverage_strata.{target,distractor,none,unseen}` |
| Table 7, easy row | `rank_report_easy.json` | `gate1_primary` |
| Table 7, other location | `rank_report.json` | `other_location_vs_base`, `gate2_other_location_control` |
| Table 7, random / degree-matched | `controls_specter2.json` | `arms.*.vs_base` |
| Table 7, raw co-count / node2vec / sibling titles / leaky graph | `rank_report_{count,node2vec,siblings,fullgraph}.json` | `gate1_primary` |
| Figure 3 | `expb_report_selfseed.json` | `information_state` (`n_anchors`, `n_correct`, `p_next_correct`) |
| Figure 4 | `anchor_map_specter2.json`, `anchor_map_specter2.pgf.txt` | `xe`, `ye`, `grid` |
| §3.2 anchor routes, §3.3 counts | `build_report_test.json` | `anchor_ge1_{clustering,marker,union}`, `alignment_failure` |
| §4.1 graph size and leakage | `graph_report.json` | `papers_counted`, `directed_nonzeros`, `held_out_papers`, `leakage_overlap` |
| §5.3 first-stage recall | `retrieve_report_test.json` | `first_stage_recall`, `gold_total` |
| §6.1 first-pick hit rate and Recall@20 by seed correctness | `expb_report_selfseed.json`, `expb_report_corpuswide.json` | `seed_quality` |
| §6.2 categories, topical split, coverage within redundant | `anchor_analysis_specter2.json` | `categories`, `topical_split`, `nulls` |
| Appendix C, seed-excluded recall (3.5× / 2.0×) | `expb_report_revealed.json` | `remaining`, `remaining_deltas`, `remaining_by_set_size` |
| docs/reproduce.md Tier-2 corpus-wide expectation | `expb_report_revealed_corpuswide.json` | `remaining`, `remaining_gold_reachability` |
| Appendix, node2vec Swap grid | `swap_report_node2vec.json` | `val_grid` |
| Appendix, anchor influence by \|A\| | `influence_specter2.json` | `by_n_anchors` |
| λ / α selection, one sweep per base or association measure | `sweep_val*.json` | `best`, `grid` |

Not here: Figure 2 and the coherence numbers in §1.3 and §5.1, which were
produced outside the harness (paper Appendix "Provenance of the coherence
figures").

Only runs that back a number in the paper are kept: the aggregator ablations
(`--agg masked_mean` / `max`) and the node2vec `(p, q)` grid other than the
selected `sweep_val_node2vec.json` are omitted; `docs/reproduce.md` lists how
to rerun them.

`expb_report_*.json` were produced with `--standardize zscore` (the default)
and 1000 bootstrap resamples; `rank_report_fullgraph.json` is the
deliberately leaky control and `graph_report_full.json` its build report.
