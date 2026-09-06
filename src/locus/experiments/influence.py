"""Counterfactual anchor influence: leave-one-anchor-out on LOCUS-Rank.

For a target c at location l with anchor set A,

    Delta_{a->c} = S(c | l, A) - S(c | l, A \\ {a})

is the change in the target's score when one observed citation is withheld.
This is reported as **model-based anchor influence / counterfactual
sensitivity**, never as a causal effect in human citation behaviour: the
intervention is on the information state handed to the recommender, not on
what an author did.

Two influences are recorded per anchor and they are not interchangeable.

  score   the closed form below. Cheap, and monotone in PPMI(c,a), so on its
          own it mostly restates the association measure.
  rank    the change in the target's reciprocal rank and hit@1 once the WHOLE
          pool is rescored without a. This is the one that can surprise:
          withdrawing an anchor lowers the distractors' scores too, so a
          target can lose score and gain rank.

The score influence has a closed form worth knowing, because it explains the
shape of the distribution. With n = |A|, p_a = PPMI(c,a) and m = mean over A,

    S(c|A) - S(c|A\\{a}) = lam * ( m - (n*m - p_a)/(n-1) )
                        = lam * (p_a - m) / (n - 1)          for n >= 2

so an anchor helps the target exactly to the degree it is MORE co-cited with
it than the average anchor is, damped by the size of the set. An anchor of
merely average association has zero influence however strong that association
is -- the mean normalisation makes influence relative, not absolute. For
n = 1 the reduced set is empty, mean_over returns 0.0, and the influence is
the whole lam * p_a; those rows are reported separately because they measure
"this anchor vs no anchors", which is a different question.

Usage:
    .venv/bin/python -m locus.experiments.influence \\
        --embeddings work/embeddings_test.parquet --sweep work/sweep_val.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
from pathlib import Path

from locus import config
from locus.core.probe import hit_at_1, reciprocal_rank
from locus.core.types import gold_sets
from locus.data.corpus import load_index
from locus.eval.sweep import build_pools
from locus.scoring.cocitation import CoGraph
from locus.scoring.node2vec import ASSOC_CHOICES, association

# Deciles are computed on the observed distribution rather than on a fixed
# grid: PPMI is heavily zero-inflated and a fixed grid would put most of the
# mass in one cell.
N_BINS = 10
MIN_BIN = 100


def rescored(pool: dict, ppmi, anchors: frozenset[str], lam: float) -> dict:
    return {c: pool["z"][c] + lam * ppmi.mean_over(c, anchors)
            for c in pool["z"]}


def influence_rows(pool: dict, ppmi, lam: float, sims=None) -> list[dict]:
    """One row per anchor: what withholding that anchor does to the target.

    `sims` is an optional callable (a, b) -> cosine, used only to record a
    covariate. It must not affect the intervention.
    """
    anchors = pool["anchors"]
    if not anchors:
        return []
    gold = pool["gold"]
    full = rescored(pool, ppmi, anchors, lam)
    rr_full = reciprocal_rank(full, gold)
    hit_full = hit_at_1(full, gold)
    n = len(anchors)

    rows = []
    for a in sorted(anchors):
        reduced_set = anchors - {a}
        reduced = rescored(pool, ppmi, reduced_set, lam)
        rows.append({
            "context_id": pool["context_id"],
            "citing_id": pool["citing_id"],
            "gold": gold,
            "anchor": a,
            "n_anchors": n,
            "ppmi": ppmi.value(gold, a),
            "d_score": full[gold] - reduced[gold],
            "d_rr": rr_full - reciprocal_rank(reduced, gold),
            "d_hit": hit_full - hit_at_1(reduced, gold),
            "sim": None if sims is None else sims(gold, a),
        })
    return rows


def _quantiles(values: list[float]) -> dict:
    xs = sorted(values)
    if not xs:
        return {}

    def q(p: float) -> float:
        return xs[min(len(xs) - 1, int(p * len(xs)))]

    return {
        "n": len(xs), "mean": statistics.fmean(xs),
        "p01": q(0.01), "p10": q(0.10), "p25": q(0.25), "median": q(0.50),
        "p75": q(0.75), "p90": q(0.90), "p99": q(0.99),
        "frac_positive": sum(x > 0 for x in xs) / len(xs),
        "frac_zero": sum(x == 0 for x in xs) / len(xs),
        "frac_negative": sum(x < 0 for x in xs) / len(xs),
    }


def _binned(rows: list[dict], key: str, field: str = "d_rr") -> list[dict]:
    """Mean influence per decile of `key`, dropping thin bins."""
    have = [r for r in rows if r.get(key) is not None]
    if not have:
        return []
    have.sort(key=lambda r: r[key])
    out, step = [], max(1, len(have) // N_BINS)
    for i in range(0, len(have), step):
        chunk = have[i:i + step]
        if len(chunk) < MIN_BIN:
            continue
        out.append({
            "bin_lo": chunk[0][key], "bin_hi": chunk[-1][key],
            "n": len(chunk),
            f"mean_{field}": statistics.fmean(r[field] for r in chunk),
        })
    return out


def _grouped(rows: list[dict], key: str, field: str = "d_rr") -> list[dict]:
    buckets: dict[int, list[float]] = {}
    for r in rows:
        if r.get(key) is not None:
            buckets.setdefault(int(r[key]), []).append(r[field])
    return [{key: k, "n": len(v), f"mean_{field}": statistics.fmean(v)}
            for k, v in sorted(buckets.items()) if len(v) >= MIN_BIN]


def asymmetry(rows: list[dict]) -> dict:
    """Compare Delta_{a->c} against Delta_{c->a} where both were measured.

    PPMI is symmetric, so any asymmetry comes from the intervention rather
    than the association: the mean over A differs between the two directions,
    the anchor sets have different sizes, and the two targets sit in different
    pools against different distractors. Pairs are keyed by citing paper so
    two unrelated papers reusing the same reference are never matched up.
    """
    by_pair: dict[tuple[str, str, str], dict[str, float]] = {}
    for r in rows:
        c, a = r["gold"], r["anchor"]
        lo, hi = (c, a) if c < a else (a, c)
        slot = by_pair.setdefault((r["citing_id"], lo, hi), {})
        slot["fwd" if c == lo else "rev"] = r["d_rr"]
    both = [(v["fwd"], v["rev"]) for v in by_pair.values()
            if "fwd" in v and "rev" in v]
    if not both:
        return {"pairs": 0}
    diffs = [abs(x - y) for x, y in both]
    same_sign = sum((x > 0) == (y > 0) for x, y in both)
    mx = statistics.fmean(x for x, _ in both)
    my = statistics.fmean(y for _, y in both)
    sx = statistics.pstdev([x for x, _ in both])
    sy = statistics.pstdev([y for _, y in both])
    cov = statistics.fmean((x - mx) * (y - my) for x, y in both)
    return {
        "pairs": len(both),
        "mean_abs_difference": statistics.fmean(diffs),
        "median_abs_difference": sorted(diffs)[len(diffs) // 2],
        "frac_same_sign": same_sign / len(both),
        "correlation": (cov / (sx * sy)) if sx > 0 and sy > 0 else None,
    }


def _clustered_at_location(split: str) -> dict[str, int]:
    """|G(l)| by the CLUSTERING route only, target included.

    This is NOT the anchor-set size and must not be reported as one. Two
    differences, both measured on test:

      * it counts by the clustering route alone, while A(l) is the union of
        the clustering and marker routes, so marker-only anchors are missing
        from it;
      * it includes the target itself, while A(l) excludes it.

    The consequence is n_anchors >= |G(l)| - 1 for every one of the 143,494
    interventions, with equality in 58.89% and a marker-route surplus in the
    remaining 41.11%. `n_anchors` is the true count of co-present anchors and
    is the field any set-size statistic should use; this one exists only to
    let the two extraction routes be compared.

    (Shipped as `set_size` before 2026-08-26; renamed because the old name
    invited exactly the confusion it is documented against here.)
    """
    index = load_index(config.WORK_DIR / f"index_{split}.pkl")
    out: dict[str, int] = {}
    for recs in index.values():
        golds = gold_sets(recs)
        for rec in recs:
            out[rec.context_id] = len(golds.get(rec.location, ()))
    return out


def _similarity(prefix: Path):
    """Cosine between two papers from the corpus-wide mmap store, or None.

    The Rank embeddings only cover pool candidates; anchors are not pool
    members and are frequently absent from them. The corpus-wide store holds
    every prefetch candidate across the split, which is where the anchors
    live. Coverage is measured and reported rather than assumed.
    """
    if not Path(f"{prefix}_index.json").exists():
        print(f"no similarity store at {prefix}_index.json; "
              f"the cosine covariate will be empty", file=sys.stderr)
        return None, 0
    from locus.scoring.mmapstore import MmapScorer
    store = MmapScorer(prefix)

    def sim(a: str, b: str):
        ia, ib = store._doc_ix.get(a), store._doc_ix.get(b)
        if ia is None or ib is None:
            return None
        va = store._doc[ia].astype("float32")
        vb = store._doc[ib].astype("float32")
        return float(va @ vb)

    return sim, len(store._doc_ix)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test")
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--sweep", type=Path,
                    default=config.WORK_DIR / "sweep_val.json")
    ap.add_argument("--graph", type=Path,
                    default=config.WORK_DIR / "cocitation_train.npz")
    # The mmap prefix, not the shard directory: mmapstore.build writes
    # "<prefix>_mm_{paper,context}.npy" alongside "<prefix>_mm_index.json".
    ap.add_argument("--sim-store", type=Path,
                    default=config.WORK_DIR / "embeddings_expb_test_mm")
    ap.add_argument("--tag", default="specter2")
    ap.add_argument("--assoc", choices=ASSOC_CHOICES, default="ppmi",
                    help="association measure; must match the --sweep used, "
                         "since lambda is not transferable across measures")
    ap.add_argument("--vectors", type=Path, default=None,
                    help="node2vec prefix, for --assoc node2vec")
    ap.add_argument("--dump", type=Path, default=None,
                    help="optional pickle of every row, for ad-hoc slicing")
    args = ap.parse_args(argv)

    sel = json.loads(args.sweep.read_text())["best"]
    lam = sel["lambda"]
    graph = CoGraph.load(args.graph)
    ppmi = association(args.assoc, graph, sel["alpha"], sel["min_count"],
                       args.vectors)
    pools = build_pools(args.split, args.embeddings)
    sizes = _clustered_at_location(args.split)
    sims, sim_vocab = _similarity(args.sim_store)
    print(f"{len(pools)} pools | lambda={lam} | "
          f"similarity store: {sim_vocab or 'absent'}")

    rows: list[dict] = []
    for p in pools:
        for r in influence_rows(p, ppmi, lam, sims):
            r["n_clustered_at_location"] = sizes.get(p["context_id"])
            j = graph.index.get(r["anchor"])
            r["degree"] = None if j is None else int(graph.marginals[j])
            rows.append(r)
    multi = [r for r in rows if r["n_anchors"] >= 2]
    single = [r for r in rows if r["n_anchors"] == 1]
    covered = sum(r["sim"] is not None for r in rows)
    print(f"{len(rows)} anchor interventions "
          f"({len(multi)} with |A|>=2, {len(single)} with |A|=1)")
    print(f"semantic-similarity covariate available for {covered} "
          f"({covered / max(1, len(rows)):.2%})")

    report = {
        "tag": args.tag, "split": args.split, "selected": sel,
        "pools": len(pools), "interventions": len(rows),
        "sim_coverage": covered / max(1, len(rows)),
        "d_rr_multi": _quantiles([r["d_rr"] for r in multi]),
        "d_rr_single": _quantiles([r["d_rr"] for r in single]),
        "d_score_multi": _quantiles([r["d_score"] for r in multi]),
        "d_hit_multi": _quantiles([r["d_hit"] for r in multi]),
        "by_ppmi": _binned(multi, "ppmi"),
        "by_similarity": _binned(multi, "sim"),
        "by_degree": _binned(multi, "degree"),
        "by_n_anchors": _grouped(rows, "n_anchors"),
        "by_n_clustered_at_location": _grouped(rows, "n_clustered_at_location"),
        "asymmetry": asymmetry(multi),
    }

    q = report["d_rr_multi"]
    print(f"\nd_rr with |A|>=2: mean {q['mean']:+.6f}  median {q['median']:+.6f}"
          f"  [{q['p01']:+.4f}, {q['p99']:+.4f}]")
    print(f"  positive {q['frac_positive']:.3%}  zero {q['frac_zero']:.3%}"
          f"  negative {q['frac_negative']:.3%}")
    print(f"\n{'ppmi decile':>26}  {'n':>7}  mean d_rr")
    for b in report["by_ppmi"]:
        print(f"  [{b['bin_lo']:>9.4f}, {b['bin_hi']:>9.4f}]  "
              f"{b['n']:>7}  {b['mean_d_rr']:+.6f}")
    if report["by_similarity"]:
        print(f"\n{'cosine decile':>26}  {'n':>7}  mean d_rr")
        for b in report["by_similarity"]:
            print(f"  [{b['bin_lo']:>9.4f}, {b['bin_hi']:>9.4f}]  "
                  f"{b['n']:>7}  {b['mean_d_rr']:+.6f}")
    print(f"\n{'|A|':>5}  {'n':>8}  mean d_rr")
    for b in report["by_n_anchors"]:
        print(f"{b['n_anchors']:>5}  {b['n']:>8}  {b['mean_d_rr']:+.6f}")
    a = report["asymmetry"]
    if a["pairs"]:
        print(f"\nasymmetry over {a['pairs']} reciprocal pairs: "
              f"corr {a['correlation']:.4f}, same sign "
              f"{a['frac_same_sign']:.3%}, mean |difference| "
              f"{a['mean_abs_difference']:.6f}")

    out = config.WORK_DIR / f"influence_{args.tag}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")
    if args.dump:
        with open(args.dump, "wb") as f:
            pickle.dump(rows, f)
        print(f"wrote {args.dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
