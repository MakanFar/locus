"""What makes an anchor informative? The PPMI x cosine grid and its strata.

The leave-one-out pass (`influence.py`) answers *how much* each anchor moves
the target. This module asks *what kind* of anchor does it, using only columns
that pass already carries -- PPMI(c,a), cosine(c,a), and the rank change --
plus one join back to the anchor sets.

Four outputs, in the order the argument runs:

  A  grid          PPMI x cosine cells with count, mean d_rr and mean d_score.
                   Two cells carry the thesis: SEMANTIC-ONLY (cosine high,
                   PPMI zero -- topically close, structurally silent) and
                   SURPRISING (cosine low, PPMI high -- structure capturing a
                   relation the embedding misses).
  B  base rates    the count and mean influence of the surprising cell, printed
                   before anything is written about it. The decile result --
                   influence rising with similarity, -0.008 to +0.036 --
                   predicts this cell is rare and weak, so it is measured
                   before it is featured, not after.
  D  nulls         the 84.27% of interventions that change nothing, split into
                   sparsity / redundancy / sub-threshold. Only the second is
                   redundancy; reporting all three as redundancy would claim
                   the set is over-determined where the real cause is that the
                   graph has nothing to say.
  F  strata        a fixed-seed stratified sample for the qualitative pass,
                   drawn from strata defined here rather than after looking.

Not covered here, deliberately: the direction asymmetry. With the mean
aggregator Delta(a->b) - Delta(b->a) is dominated by a normalisation
difference between the two anchor sets, so reading it as a directional
citation relation would be reading the estimator, not the data.

Usage:
    .venv/bin/python -m locus.experiments.anchor_analysis --tag specter2
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import random
import statistics
import sys
from pathlib import Path

from locus import config

# Terciles keep the grid readable and keep every cell populated; the influence
# report already carries the decile view of each axis on its own.
COSINE_BANDS = ("low", "mid", "high")
PPMI_BANDS = ("zero", "low", "high")


def _tercile_edges(values: list[float]) -> tuple[float, float]:
    """Lower and upper tercile cut points, or (0, 0) for an empty band.

    The empty case is not hypothetical for `grid`: it is reached whenever no
    pair in the input has non-zero PPMI, which is what a degenerate slice or a
    graph built with too high a min_count looks like. Raising there would turn
    a legitimately empty measurement into a crash halfway through a report.
    """
    xs = sorted(values)
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    return xs[n // 3], xs[min(2 * n // 3, n - 1)]


def band_ppmi(v: float, hi: float) -> str:
    """Zero is its own band, not the bottom of a continuum.

    Roughly 60% of anchor-target pairs never co-occur in the training graph,
    so a tercile split would put the zeros across the bottom two bands and
    hide the one distinction the analysis is about: no evidence versus weak
    evidence.
    """
    if v <= 0.0:
        return "zero"
    return "high" if v >= hi else "low"


def band_cosine(v: float, lo: float, hi: float) -> str:
    if v < lo:
        return "low"
    return "high" if v >= hi else "mid"


def grid(rows: list[dict]) -> dict:
    """Per-cell count, mean rank influence and mean score influence."""
    sims = [r["sim"] for r in rows if r["sim"] is not None]
    lo, hi = _tercile_edges(sims)
    pos = [r["ppmi"] for r in rows if r["ppmi"] > 0.0]
    _, p_hi = _tercile_edges(pos)

    cells: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for r in rows:
        if r["sim"] is None:
            continue
        cells[(band_ppmi(r["ppmi"], p_hi), band_cosine(r["sim"], lo, hi))
              ].append(r)

    out = {
        "cosine_edges": [lo, hi],
        "ppmi_high_edge": p_hi,
        "n_with_similarity": len(sims),
        "cells": {},
    }
    for p in PPMI_BANDS:
        for c in COSINE_BANDS:
            rs = cells.get((p, c), [])
            out["cells"][f"ppmi_{p}|cos_{c}"] = {
                "n": len(rs),
                "frac": len(rs) / max(len(sims), 1),
                "mean_d_rr": statistics.fmean(x["d_rr"] for x in rs) if rs else None,
                "mean_d_score": statistics.fmean(x["d_score"] for x in rs) if rs else None,
                "frac_d_rr_positive": (
                    sum(1 for x in rs if x["d_rr"] > 0) / len(rs) if rs else None),
            }
    return out


def null_decomposition(rows: list[dict]) -> dict:
    """Split the zero-influence interventions by cause.

    A null tells you nothing on its own. Three quite different situations
    produce one:

      sparsity       PPMI(c,a) = 0 -- the graph has no relation to lose.
      redundancy     PPMI(c,a) > 0 and a co-present sibling still carries the
                     evidence after removal.
      sub-threshold  PPMI(c,a) > 0, no covering sibling, and removal simply
                     did not cross a rank boundary.

    Covering is reported under two definitions because the word is doing real
    work. `relative` asks whether at least as much evidence survives as was
    removed, which is self-calibrating and threshold-free. `absolute` asks
    whether a surviving sibling clears the global top-PPMI-decile bar, which
    is stricter and does not count a weak anchor removed from a weak set.
    """
    by_target: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_target[(r["context_id"], r["gold"])].append(r)

    pos = sorted(r["ppmi"] for r in rows if r["ppmi"] > 0.0)
    decile10 = pos[int(0.9 * len(pos))] if pos else 0.0

    counts = {k: 0 for k in ("sparsity", "redundancy_relative",
                             "redundancy_absolute", "subthreshold_relative",
                             "subthreshold_absolute")}
    n_null = 0
    for rs in by_target.values():
        ppmis = [r["ppmi"] for r in rs]
        for i, r in enumerate(rs):
            if r["d_rr"] != 0.0:
                continue
            n_null += 1
            if r["ppmi"] <= 0.0:
                counts["sparsity"] += 1
                continue
            siblings = ppmis[:i] + ppmis[i + 1:]
            rel = any(s >= r["ppmi"] for s in siblings)
            absolute = any(s >= decile10 for s in siblings)
            counts["redundancy_relative" if rel else
                   "subthreshold_relative"] += 1
            counts["redundancy_absolute" if absolute else
                   "subthreshold_absolute"] += 1
    return {
        "n_interventions": len(rows),
        "n_null": n_null,
        "frac_null": n_null / max(len(rows), 1),
        "ppmi_decile10_edge": decile10,
        "counts": counts,
        "fractions_of_null": {k: v / max(n_null, 1) for k, v in counts.items()},
    }


CATEGORIES = ("informative", "redundant", "non_cocited")


def categorise(rows: list[dict]) -> dict[str, list[dict]]:
    """Partition every intervention into three exhaustive categories.

      informative   PPMI > 0 and the removal costs rank (d_rr > 0)
      redundant     PPMI > 0 and the removal costs nothing (d_rr <= 0)
      non_cocited   PPMI = 0 -- no learned relation to lose

    `redundant` is deliberately broader than "a sibling covers it". Requiring
    coverage leaves ~9% of interventions in neither category, and a taxonomy
    that silently drops a tenth of its population is not a taxonomy. Coverage
    is reported as a property *within* the category instead, where it belongs:
    72.17% of redundant removals leave a sibling with at least as much
    evidence, and the rest are sub-threshold -- real but too small to cross a
    rank boundary.

    There is deliberately no fourth "surprising" category for high-PPMI,
    low-cosine anchors. Their cosine distribution inside `informative` is
    unimodal (a single mode at 0.938) and the topically-aligned and
    cross-topic halves differ in mean influence by 0.016 on values near 0.41,
    so the split is a slice through a continuum, not a mode. It is reported as
    a sub-split in `topical_split`, not as a type.
    """
    by_target: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_target[(r["context_id"], r["gold"])].append(r)
    out: dict[str, list[dict]] = {k: [] for k in CATEGORIES}
    for rs in by_target.values():
        for r in rs:
            if r["ppmi"] <= 0.0:
                out["non_cocited"].append(r)
            elif r["d_rr"] > 0.0:
                out["informative"].append(r)
            else:
                out["redundant"].append(r)
    return out


def topical_split(informative: list[dict], edge: float) -> dict:
    """Informative anchors either side of a cosine cut, with the modality check.

    The number that matters is the mean-influence gap: if the two halves are
    two phenomena it should be large, and it is not.
    """
    known = [r for r in informative if r["sim"] is not None]
    hi = [r for r in known if r["sim"] >= edge]
    lo = [r for r in known if r["sim"] < edge]
    counts, _ = _histogram([r["sim"] for r in known])
    modes = _count_modes(counts)
    def side(rs):
        return {"n": len(rs),
                "mean_d_rr": statistics.fmean(r["d_rr"] for r in rs) if rs
                else None}

    return {
        "edge": edge, "n_known_similarity": len(known), "modes": modes,
        "aligned": side(hi), "cross_topic": side(lo),
    }


def _count_modes(counts: list[int], prominence: float = 0.05) -> int:
    """Local maxima of a 3-bin-smoothed histogram, boundaries included.

    Two details decide whether "the distribution is unimodal" is a statement
    about the data or about this function.

    Boundaries count. An earlier version scanned only interior indices, so a
    distribution with both of its peaks near the ends of the range reported
    ZERO modes -- the most confidently wrong answer available. A bin at either
    end is a peak if it exceeds its single neighbour.

    Tiny bumps do not. A peak must reach `prominence` of the tallest bin,
    which stops a one-count wobble in a sparse tail from being reported
    alongside a mode carrying thousands of observations.
    """
    if not any(counts):
        return 0
    sm = [statistics.fmean(counts[max(0, i - 1):i + 2])
          for i in range(len(counts))]
    floor = prominence * max(sm)
    modes = 0
    for i, v in enumerate(sm):
        if v < floor:
            continue
        left = sm[i - 1] if i else float("-inf")
        right = sm[i + 1] if i + 1 < len(sm) else float("-inf")
        if (v > left and v >= right and v != right) or (
            v > left and i + 1 == len(sm)
        ):
            modes += 1
    return modes


def _histogram(values: list[float], bins: int = 20) -> tuple[list[int], float]:
    """Equal-width counts, or a single occupied bin for a degenerate range.

    A width of zero is reachable whenever every value is identical -- one
    observation, or a stratum in which every pair scored the same cosine --
    and dividing by it would take down a report over a case that has an
    obvious answer: one bin, one mode.
    """
    if not values:
        return [0] * bins, 0.0
    lo, hi = min(values), max(values)
    w = (hi - lo) / bins
    counts = [0] * bins
    if w == 0.0:
        counts[0] = len(values)
        return counts, 0.0
    for v in values:
        counts[min(int((v - lo) / w), bins - 1)] += 1
    return counts, w


PPMI_EDGES = (0.0, 1e-9, 7.5, 8.5, 9.0, 9.5, 10.0, 11.0)
"""Lower edges of the PPMI bands in the paper's anchor map (Figure 4).

Zero is its own band, closed at 1e-9 so that PPMI == 0 never leaks into the
first positive band; the remaining edges are fixed rather than quantiles so
the same band means the same evidence strength whichever base produced the
rows. The top band is closed above at max(PPMI) + 1 in `anchor_map`.
"""

N_OCTILES = 8


def anchor_map(rows: list[dict]) -> dict:
    """Mean rank influence on a PPMI-band x cosine-octile grid (Figure 4).

    Rows without a cosine are excluded, as in `grid`. The cosine edges are the
    octiles of the observed similarities (numpy's default linear quantile), so
    every column carries about an eighth of the interventions; the PPMI edges
    are `PPMI_EDGES`. Both the top PPMI band and the rightmost octile are
    closed, so the maximum of either axis lands in a cell rather than falling
    off the grid.

    Returns `{"xe": [9 cosine edges], "ye": [9 PPMI edges],
    "grid": [(band_lo, band_hi, [(n, mean_d_rr) x 8]), ...]}` with bands
    listed top-down (highest PPMI first), which is the order the figure draws.
    """
    import numpy as np

    have = [r for r in rows if r["sim"] is not None]
    sims = np.asarray([r["sim"] for r in have], dtype=float)
    xe = [float(v) for v in np.quantile(sims, np.linspace(0.0, 1.0, N_OCTILES + 1))]
    ye = [*PPMI_EDGES, max(r["ppmi"] for r in have) + 1.0]

    def column(sim: float) -> int:
        for xi in range(N_OCTILES):
            closed = xi == N_OCTILES - 1
            if xe[xi] <= sim < xe[xi + 1] or (closed and sim <= xe[xi + 1]):
                return xi
        raise ValueError(f"cosine {sim!r} outside [{xe[0]}, {xe[-1]}]")

    def band(ppmi: float) -> int:
        for yi in range(len(ye) - 1):
            if ye[yi] <= ppmi < ye[yi + 1]:
                return yi
        raise ValueError(f"ppmi {ppmi!r} outside [{ye[0]}, {ye[-1]})")

    cells: dict[tuple[int, int], list[float]] = collections.defaultdict(list)
    for r in have:
        cells[(band(r["ppmi"]), column(r["sim"]))].append(r["d_rr"])

    grid_rows = []
    for yi in range(len(ye) - 2, -1, -1):
        row = []
        for xi in range(N_OCTILES):
            vals = cells.get((yi, xi), [])
            row.append((len(vals), statistics.fmean(vals) if vals else None))
        grid_rows.append((ye[yi], ye[yi + 1], row))
    return {"xe": xe, "ye": ye, "grid": grid_rows}


def pgfplots_table(anchor_map_: dict) -> str:
    """The `x y value` block figure_4.tex reads, bands top-down, 4 dp.

    x is the cosine octile 0..7, y the PPMI band 0..7 with 0 the zero band and
    7 the >= 11 band. An empty cell prints `nan`, which pgfplots leaves blank.
    """
    n_bands = len(anchor_map_["grid"])
    blocks = []
    for i, (_lo, _hi, cells) in enumerate(anchor_map_["grid"]):
        y = n_bands - 1 - i
        blocks.append("\n".join(
            f"{x} {y} {'nan' if mean is None else f'{mean:.4f}'}"
            for x, (_n, mean) in enumerate(cells)))
    return "\n\n".join(blocks)


STRATA = ("informative", "redundant", "non_cocited")


def strata(rows: list[dict], edge: float, seed: int = config.SEED,
           per_stratum: int = 10) -> dict[str, dict]:
    """Fixed-seed stratified sample, with the strata fixed before looking.

    Four strata, one per row of the taxonomy plus the topical sub-split of
    `informative`, so a shown example can always be placed in the partition
    rather than chosen for being striking.

    Rows without a cosine are excluded from every stratum, not just the
    cosine-defined ones. 64 of 143,494 interventions (0.045%, 51 distinct
    papers) have no vector, because the similarity store covers the union of
    the split's prefetch lists and those papers were never proposed by any
    prefetch. They cannot be recomputed without embedding papers outside that
    union, and an example printed with a blank cosine is not an example of
    anything on the cosine axis -- so they are dropped here and reported.
    """
    cats = categorise(rows)
    by_target: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for r in rows:
        by_target[(r["context_id"], r["gold"])].append(r["ppmi"])

    def covered(r) -> bool:
        ps = by_target[(r["context_id"], r["gold"])]
        return sum(1 for s in ps if s >= r["ppmi"]) > 1

    def known(rs):
        return [r for r in rs if r["sim"] is not None]

    inf = known(cats["informative"])
    pools = {
        "informative_aligned": [r for r in inf if r["sim"] >= edge],
        "informative_cross_topic": [r for r in inf if r["sim"] < edge],
        "redundant": [r for r in known(cats["redundant"]) if covered(r)],
        "non_cocited_harmful": [r for r in known(cats["non_cocited"])
                                if r["d_rr"] < 0.0],
    }
    rng = random.Random(seed)
    out = {}
    for k, pool in pools.items():
        pool = sorted(pool, key=lambda r: (r["context_id"], r["anchor"]))
        out[k] = {
            "population": len(pool),
            "mean_d_rr": statistics.fmean(r["d_rr"] for r in pool) if pool else None,
            "sample": rng.sample(pool, min(per_stratum, len(pool))),
        }
    out["_excluded_no_similarity"] = sum(1 for r in rows if r["sim"] is None)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="specter2")
    ap.add_argument("--rows", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--per-stratum", type=int, default=10)
    args = ap.parse_args(argv)

    rows_path = args.rows or (
        config.WORK_DIR / f"influence_rows_{args.tag}.pkl")
    out = args.out or (
        config.WORK_DIR / f"anchor_analysis_{args.tag}.json")
    with open(rows_path, "rb") as f:
        rows = pickle.load(f)
    print(f"{len(rows)} interventions from {rows_path}")

    g = grid(rows)
    print(f"\ncosine terciles at {g['cosine_edges'][0]:.4f} / "
          f"{g['cosine_edges'][1]:.4f}; PPMI high band >= "
          f"{g['ppmi_high_edge']:.4f}")
    print(f"\n{'cell':<24}{'n':>8}{'frac':>8}{'mean d_rr':>12}{'d_rr>0':>9}")
    for k, v in g["cells"].items():
        if not v["n"]:
            print(f"{k:<24}{0:>8}")
            continue
        print(f"{k:<24}{v['n']:>8}{v['frac']:>8.3f}"
              f"{v['mean_d_rr']:>+12.5f}{v['frac_d_rr_positive']:>9.3f}")

    # B: the base-rate check, printed before the cell is written about.
    surp = g["cells"]["ppmi_high|cos_low"]
    sem = g["cells"]["ppmi_zero|cos_high"]
    print(f"\nSURPRISING (ppmi high, cosine low): n={surp['n']} "
          f"({surp['frac']:.4f} of pairs), mean d_rr {surp['mean_d_rr']:+.5f}")
    print(f"SEMANTIC-ONLY (ppmi zero, cosine high): n={sem['n']} "
          f"({sem['frac']:.4f}), mean d_rr {sem['mean_d_rr']:+.5f}")

    nulls = null_decomposition(rows)
    print(f"\nnull interventions: {nulls['n_null']} "
          f"({nulls['frac_null']:.4f} of {nulls['n_interventions']})")
    for k, v in nulls["fractions_of_null"].items():
        print(f"  {k:<26}{nulls['counts'][k]:>8}  {v:.4f}")

    cats = categorise(rows)
    print(f"\n{'category':<14}{'n':>8}{'frac':>8}{'mean d_rr':>12}{'mean PPMI':>11}")
    for k in CATEGORIES:
        rs = cats[k]
        print(f"  {k:<12}{len(rs):>8}{len(rs) / len(rows):>8.4f}"
              f"{statistics.fmean(x['d_rr'] for x in rs):>+12.5f}"
              f"{statistics.fmean(x['ppmi'] for x in rs):>11.3f}")
    assert sum(len(v) for v in cats.values()) == len(rows), "not a partition"

    ts = topical_split(cats["informative"], g["cosine_edges"][0])
    print(f"\ntopical sub-split of informative at cosine {ts['edge']:.4f} "
          f"({ts['modes']} mode(s) in the cosine histogram):")
    print(f"  aligned     n={ts['aligned']['n']:>6} "
          f"mean d_rr {ts['aligned']['mean_d_rr']:+.4f}")
    print(f"  cross-topic n={ts['cross_topic']['n']:>6} "
          f"mean d_rr {ts['cross_topic']['mean_d_rr']:+.4f}")

    st = strata(rows, g["cosine_edges"][0], per_stratum=args.per_stratum)
    print(f"\nstrata (excluding {st['_excluded_no_similarity']} rows with no "
          "similarity):")
    for k, v in st.items():
        if k.startswith("_"):
            continue
        print(f"  {k:<26}{v['population']:>8}  {v['mean_d_rr']:+.5f}  "
              f"sampled {len(v['sample'])}")

    report = {"tag": args.tag, "grid": g, "nulls": nulls,
              "categories": {k: {
                  "n": len(v), "frac": len(v) / len(rows),
                  "mean_d_rr": statistics.fmean(x["d_rr"] for x in v),
                  "mean_ppmi": statistics.fmean(x["ppmi"] for x in v),
              } for k, v in cats.items()},
              "topical_split": ts, "strata": st}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")

    # Figure 4. Written as its own file, in the shape the figure was first
    # produced from, so the paper's grid can be diffed against it directly.
    amap = anchor_map(rows)
    map_path = out.with_name(f"anchor_map_{args.tag}.json")
    map_path.write_text(json.dumps(amap, indent=1), encoding="utf-8")
    tex_path = out.with_name(f"anchor_map_{args.tag}.pgf.txt")
    tex_path.write_text(pgfplots_table(amap) + "\n", encoding="utf-8")
    top = amap["grid"][0][2]
    print(f"\nanchor map: PPMI >= {amap['ye'][-2]:g} band, mean d_rr by cosine "
          "octile: " + " ".join(f"{m:+.3f}" if m is not None else "  nan"
                                 for _n, m in top))
    print(f"wrote {map_path}\nwrote {tex_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
