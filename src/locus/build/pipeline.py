"""Build driver: build the frozen artefacts and run the gates.

Usage:  .venv/bin/python -m locus.build.pipeline [--split test|val]
        .venv/bin/python -m locus.build.pipeline --corpus DIR --split test
Exit code 0 iff every gate passes.

Every artefact is suffixed with the split, so `val` (which selects lambda)
never clobbers the frozen `test` pools that carry the headline numbers.

`--corpus DIR` runs the whole build against a JSONL corpus in the contract
`locus.data.ingest` documents, instead of the upstream Gu et al. corpus at
`$LOCUS_DATA_DIR`. Both sides are redirected -- contexts through
`jsonl_source` and papers through `jsonl_papers` -- because redirecting only
the first is what used to make this path dead-end at step 4 with
"LOCUS_DATA_DIR is not set". With no `--corpus`, nothing below reads a
different byte than it did before the flag existed.

**Two of the gates are calibrated to Gu et al.'s test split, not to the
benchmark definition**: the 40-85% hard-fraction band and the >= 30,000
headline floor are statements about the size and composition of one corpus.
Under `--corpus` they are reported but do not fail the run, and the report
records them under `scale_notes`. The gates that state what the benchmark IS
-- the null controls landing exactly on their floors, alignment, anchor
coverage -- are enforced identically on every corpus. Nothing about this
changes without `--corpus`.
"""
import argparse
import collections
import json
import math
import pickle
import sys

from locus import config
from locus.core.bootstrap import cluster_ci
from locus.core.hardness import is_easy
from locus.core.pools import build_rank_items, build_swap_pairs
from locus.core.types import (
    anchors,
    anchors_union,
    clustering_disagreement,
    gold_sets,
)
from locus.data.corpus import build_index, load_papers, save_index
from locus.scoring.baselines import (
    constant_scorer,
    popularity_scorer,
    rank_mrr,
    swap_accuracy,
)


def floor_ok(values, aggregate: float, floor: float) -> bool:
    """Is this scorer exactly on its chance floor?

    The load-bearing test is on the INPUTS: a context-independent constant
    scorer ties every pool, and ties are scored by their expectation, so every
    per-item reciprocal rank must equal the floor exactly. That check is
    n-independent and admits no tolerance.

    The aggregate is then allowed a few ULPs. `paper_mean` is
    `math.fsum(xs) / len(xs)`: fsum is exactly rounded, but the division is
    not, and `math.fsum([h] * n) / n != h` for 53 group sizes in 1..2000 at
    H_10/10 alone. Asserting equality on the aggregate tests the floating-point
    division, not the benchmark.
    """
    if set(values) != {floor}:
        return False
    return abs(aggregate - floor) <= 4 * math.ulp(floor)


def _record(
    failures: list[str], scale_notes: list[str], corpus: str | None, message: str
) -> None:
    """File a threshold violation as a failure, or as a note under `--corpus`.

    Only the two Gu-calibrated thresholds route through here (see the module
    docstring). With no `--corpus` this is `failures.append` and nothing else,
    so the published build's behaviour is unchanged.
    """
    (scale_notes if corpus is not None else failures).append(message)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # train is excluded on purpose: it has no pools and no gates -- the train
    # split feeds build_graph.py, which streams it instead of indexing it.
    ap.add_argument("--split", choices=("test", "val"), default="test")
    ap.add_argument(
        "--corpus", default=None,
        help="a JSONL corpus directory in the contract documented in "
             "locus.data.ingest (contexts.jsonl, papers.jsonl, splits/). "
             "Without it, the upstream corpus at $LOCUS_DATA_DIR is used and "
             "every step behaves exactly as before.",
    )
    ap.add_argument(
        "--pool-size", type=int, default=config.POOL_SIZE,
        help=f"candidates per LOCUS-Rank pool (default {config.POOL_SIZE}, the "
             "published value). A pool of k needs k-1 distractors from OTHER "
             "locations of the same citing paper, so a corpus with fewer "
             "references per paper has to lower this; the chance floor is "
             "H_k/k and moves with it.",
    )
    args = ap.parse_args(argv)
    split = args.split
    pool_size = args.pool_size

    # Both halves of the corpus, or neither. Redirecting contexts without
    # papers is exactly the bug this flag fixes.
    record_source = paper_source = None
    if args.corpus is not None:
        from locus.data.ingest import jsonl_papers, jsonl_source

        record_source = jsonl_source(args.corpus)
        paper_source = jsonl_papers(args.corpus)

    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    scale_notes: list[str] = []

    where = args.corpus or "contexts.json"
    print(f"[1/6] building {split} index (one pass over {where}, ~8 min)")
    index = build_index(split, source=record_source)
    save_index(index, config.WORK_DIR / f"index_{split}.pkl")
    n_ctx = sum(len(v) for v in index.values())
    print(f"      {n_ctx} contexts across {len(index)} citing papers")
    if n_ctx == 0:
        # config.contexts() itself raises SystemExit when LOCUS_DATA_DIR is
        # unset -- constructing this message must not risk replacing the
        # "0 contexts loaded" diagnosis with a different, misleading error.
        if args.corpus is not None:
            source_desc = f"{args.corpus} (--corpus)"
            hint = "check the corpus directory and that splits/ is populated"
        else:
            try:
                source_desc = str(config.contexts())
            except SystemExit:
                source_desc = "<LOCUS_DATA_DIR unset>"
            hint = "check LOCUS_DATA_DIR and that the split file is populated"
        raise SystemExit(
            f"0 contexts loaded for split {split!r} from {source_desc} — {hint}"
        )

    print("[2/6] alignment, locations, anchor coverage")
    align_ok = sum(1 for recs in index.values() for r in recs if r.alignment_ok)
    align_bad = n_ctx - align_ok
    # Split the failure: a parse failure means markers_with_numbers gave up; a
    # validation failure means it produced a number the bibliography contradicts.
    # They have different causes and different acceptable rates.
    parse_bad = sum(
        1
        for recs in index.values()
        for r in recs
        if not r.alignment_ok and not r.marker_anchors and r.n_unresolvable == 0
        and r.n_implausible == 0 and r.n_self == 0
    )
    validation_bad = align_bad - parse_bad
    n_locations = sum(len({r.location for r in recs}) for recs in index.values())
    unresolvable = sum(r.n_unresolvable for recs in index.values() for r in recs)
    implausible = sum(r.n_implausible for recs in index.values() for r in recs)
    self_anchors = sum(r.n_self for recs in index.values() for r in recs)

    # anchor_counts (and the |A|>=1 gate) are computed on the UNION of the two
    # anchor routes -- clustering (anchors) and marker (rec.marker_anchors).
    # The two are not nested: on the test split, clustering-only coverage,
    # marker-only coverage and the union all differ, so all three are tracked
    # separately below. `anchors`/`clustering_disagreement` stay clustering-only
    # -- they are the THETA diagnostic, not a coverage figure.
    anchor_counts: collections.Counter = collections.Counter()
    disagree = 0
    marker_anchors_total = 0
    ge1_clustering = 0
    ge1_marker = 0
    for recs in index.values():
        golds = gold_sets(recs)
        for rec in recs:
            if anchors(rec, golds):
                ge1_clustering += 1
            if rec.marker_anchors:
                ge1_marker += 1
            marker_anchors_total += len(rec.marker_anchors)
            anchor_counts[len(anchors_union(rec, golds))] += 1
            disagree += clustering_disagreement(rec, golds)
    ge1_union = sum(v for k, v in anchor_counts.items() if k >= 1)

    print(f"      alignment: {align_ok} ok / {align_bad} failed "
          f"({100 * align_bad / n_ctx:.2f}%) "
          f"[~{parse_bad} parse, ~{validation_bad} validation]")
    print(f"      locations: {n_locations} "
          f"({n_ctx / max(n_locations, 1):.2f} contexts per location)")
    print(f"      marker anchors: {marker_anchors_total} resolved, "
          f"{self_anchors} self, {unresolvable} unresolvable, "
          f"{implausible} implausible")
    print(f"      clustering disagreement: {disagree} marker anchors outside "
          f"their own cluster ({100 * disagree / max(marker_anchors_total, 1):.2f}% "
          f"of {marker_anchors_total} marker anchors)")
    print(f"      |A|>=1 clustering: {ge1_clustering} "
          f"({100 * ge1_clustering / n_ctx:.2f}%)")
    print(f"      |A|>=1 marker:     {ge1_marker} "
          f"({100 * ge1_marker / n_ctx:.2f}%)")
    print(f"      |A|>=1 union:      {ge1_union} "
          f"({100 * ge1_union / n_ctx:.2f}%)")
    for k in range(6):
        print(f"        |A_union|=={k}: {anchor_counts[k]} "
              f"({100 * anchor_counts[k] / n_ctx:.2f}%)")
    if ge1_union / n_ctx < config.MIN_ANCHOR_COVERAGE:
        failures.append(
            f"anchor coverage (union) {100 * ge1_union / n_ctx:.1f}% < "
            f"{100 * config.MIN_ANCHOR_COVERAGE:.0f}%"
        )
    # Expect ~8-11% total: ~2-5% parse failure plus ~6% bibliography-invalid
    # targets, both measured on 600k real contexts. 15% is the alarm line.
    if align_bad / n_ctx > config.MAX_ALIGNMENT_FAILURE:
        failures.append(
            f"alignment failure {100 * align_bad / n_ctx:.1f}% > "
            f"{100 * config.MAX_ALIGNMENT_FAILURE:.0f}%"
        )

    print("[3/6] freezing pools and swap pairs")
    rank_items, pool_drops = build_rank_items(index, pool_size=pool_size)
    swap_pairs = build_swap_pairs(index)
    print(f"      {len(rank_items)} rank items, {len(swap_pairs)} swap pairs")
    for reason, n in pool_drops.most_common():
        print(f"        {reason}: {n}")

    rank_items_path = config.WORK_DIR / f"rank_items_{split}.pkl"
    swap_pairs_path = config.WORK_DIR / f"swap_pairs_{split}.pkl"
    with open(rank_items_path, "wb") as f:
        pickle.dump(rank_items, f, protocol=5)
    with open(swap_pairs_path, "wb") as f:
        pickle.dump(swap_pairs, f, protocol=5)
    print(f"      wrote {rank_items_path}")
    print(f"      wrote {swap_pairs_path}")

    print("[4/6] hardness stratification (second streaming pass, ~5 min)")
    by_ctx = {r.context_id: r for r in (x for v in index.values() for x in v)}
    papers = load_papers({it.gold for it in rank_items}, source=paper_source)
    hard_ids = set()
    for it in rank_items:
        meta = papers.get(it.gold)
        if meta is None:
            continue
        if not is_easy(
            by_ctx[it.context_id].raw,
            meta["title"],
            # `.get`: the JSONL contract makes authors optional, and a corpus
            # without it loses the surname route rather than crashing here.
            meta.get("authors"),
            config.HARDNESS_TAU,
        ):
            hard_ids.add(it.context_id)
    frac = len(hard_ids) / max(len(rank_items), 1)
    print(f"      hard: {len(hard_ids)} ({100 * frac:.2f}%)")
    lo_band, hi_band = config.HARD_FRACTION_BAND
    if not lo_band <= frac <= hi_band:
        _record(
            failures, scale_notes, args.corpus,
            f"hard fraction {100 * frac:.1f}% outside the "
            f"{100 * lo_band:.0f}-{100 * hi_band:.0f}% band",
        )

    print(f"[5/6] headline set = hard AND pool>={pool_size} AND |A|>=1 (union)")
    anchor_ok = set()
    for recs in index.values():
        golds = gold_sets(recs)
        for rec in recs:
            if anchors_union(rec, golds):
                anchor_ok.add(rec.context_id)
    headline = [
        it for it in rank_items if it.context_id in hard_ids and it.context_id in anchor_ok
    ]
    print(f"      headline set: {len(headline)}")
    if len(headline) < config.MIN_HEADLINE:
        _record(
            failures, scale_notes, args.corpus,
            f"headline set {len(headline)} < {config.MIN_HEADLINE}",
        )

    headline_ids = {it.context_id for it in headline}
    hard_ids_path = config.WORK_DIR / f"hard_ids_{split}.pkl"
    headline_ids_path = config.WORK_DIR / f"headline_ids_{split}.pkl"
    with open(hard_ids_path, "wb") as f:
        pickle.dump(sorted(hard_ids), f, protocol=5)
    with open(headline_ids_path, "wb") as f:
        pickle.dump(sorted(headline_ids), f, protocol=5)
    print(f"      wrote {hard_ids_path}")
    print(f"      wrote {headline_ids_path}")

    print("[6/6] null controls (must land on their floor EXACTLY)")
    # Counted over every context in the index, not over rank_items golds:
    # swap pairs are built from records build_rank_items drops (alignment
    # failures, and whole papers rejected as too_few_distractors), so scoping
    # the counter to rank_items leaves popularity at log1p(0)=0.0 for a third
    # of swap pairs -- byte-identical to constant_scorer(0.0) there, which
    # collapses the two controls into one.
    counts = collections.Counter(r.refid for recs in index.values() for r in recs)
    null_controls = {}

    # Swap: the lemma pins any context-independent scorer to exactly 0.5.
    for name, scorer in (
        ("popularity", popularity_scorer(counts)),
        ("constant", constant_scorer(1.0)),
    ):
        acc, outcomes, groups = swap_accuracy(swap_pairs, scorer)
        lo, hi = cluster_ci(outcomes, groups, n_boot=1000, seed=config.SEED)
        ok = floor_ok(outcomes, acc, 0.5) and floor_ok([lo, hi], lo, 0.5) and lo == hi
        null_controls[f"swap_{name}"] = {"value": acc, "ci": [lo, hi], "ok": ok}
        print(f"      swap {name}: {acc!r} CI=[{lo!r}, {hi!r}] "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(
                f"swap null control {name} = {acc!r} CI=[{lo!r}, {hi!r}], "
                "expected exactly 0.5 with a zero-width interval"
            )

    # Rank is the pre-registered primary metric, so it gets the same treatment:
    # a constant scorer ties every pool, and ties are scored by their exact
    # expectation, so MRR is exactly H_10/10.
    # H_k/k for the k actually built, not for config.POOL_SIZE: the floor MOVES
    # with k (0.29290 at 10, 0.31433 at 9), so under --pool-size the constant
    # would be the floor of pools that were never built.
    rank_floor = sum(1.0 / i for i in range(1, pool_size + 1)) / pool_size
    mrr, values, groups = rank_mrr(rank_items, constant_scorer(1.0))
    lo, hi = cluster_ci(values, groups, n_boot=1000, seed=config.SEED)
    ok = (
        floor_ok(values, mrr, rank_floor)
        and abs(lo - rank_floor) <= 4 * math.ulp(rank_floor)
        and abs(hi - rank_floor) <= 4 * math.ulp(rank_floor)
    )
    null_controls["rank_constant"] = {"value": mrr, "ci": [lo, hi], "ok": ok}
    print(f"      rank constant: {mrr!r} CI=[{lo!r}, {hi!r}] "
          f"{'OK' if ok else 'FAIL'} (floor {rank_floor!r})")
    if not ok:
        failures.append(
            f"rank null control = {mrr!r} CI=[{lo!r}, {hi!r}], "
            f"expected exactly {rank_floor!r} with a zero-width interval"
        )

    # Guard that the floor above is the tie rule and not a stuck value: unlike
    # Swap, Rank is not lemma-pinned for a varying context-independent scorer.
    pop_mrr, _, _ = rank_mrr(rank_items, popularity_scorer(counts))
    moved = pop_mrr != rank_floor
    null_controls["rank_popularity"] = {"value": pop_mrr, "moves_off_floor": moved}
    print(f"      rank popularity: {pop_mrr!r} "
          f"{'moves off the floor OK' if moved else 'STUCK ON FLOOR'}")
    if not moved:
        failures.append(
            "rank popularity control is pinned to the tie floor, so the rank "
            "floor is not evidence the tie rule is working"
        )

    report = {
        "split": split,
        "corpus": args.corpus,
        "config": {
            # The pool size ACTUALLY built with, not config.POOL_SIZE: they
            # differ only under --pool-size, and a report that named the
            # constant would describe a build that did not happen.
            "pool_size": pool_size,
            "swap_k": config.SWAP_K,
            "shingle_n": config.SHINGLE_N,
            "theta": config.THETA,
            "hardness_tau": config.HARDNESS_TAU,
            "seed": config.SEED,
        },
        "contexts": n_ctx,
        "citing_papers": len(index),
        "alignment_success": align_ok,
        "alignment_failure": align_bad,
        "alignment_parse_failure": parse_bad,
        "alignment_validation_failure": validation_bad,
        "num_locations": n_locations,
        "contexts_per_location": n_ctx / max(n_locations, 1),
        "anchor_distribution": {str(k): anchor_counts[k] for k in sorted(anchor_counts)},
        "anchor_ge1_clustering": ge1_clustering,
        "anchor_ge1_marker": ge1_marker,
        "anchor_ge1_union": ge1_union,
        "marker_anchors_self": self_anchors,
        "marker_anchors_unresolvable": unresolvable,
        "marker_anchors_implausible": implausible,
        "marker_anchors_total": marker_anchors_total,
        "clustering_disagreement": disagree,
        "pool_drop_reasons": dict(pool_drops),
        "rank_items": len(rank_items),
        "swap_pair_count": len(swap_pairs),
        "hard": len(hard_ids),
        "headline": len(headline),
        "null_controls": null_controls,
        "failures": failures,
        "scale_notes": scale_notes,
    }
    report_path = config.WORK_DIR / f"build_report_{split}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if scale_notes:
        print(
            "\nNOTE: these thresholds describe Gu et al.'s test split, not the "
            "benchmark definition, and are reported rather than enforced under "
            "--corpus:"
        )
        for note in scale_notes:
            print(f"  - {note}")
    if failures:
        print("\nGATE FAILED:")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    print("\nAll build gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
