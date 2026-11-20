"""The hard-versus-easy contrast with a joint paper-level bootstrap.

`locus eval rank` reports each slice's gain with its own interval. The paper's
appendix also compares the two gains, and a ratio or difference of two
separately bootstrapped means has no valid interval of its own. This script
scores both slices, then runs ONE paper-level bootstrap -- the same
resampling as `locus.core.bootstrap.cluster_ci` (citing papers with
replacement, per-paper means, 1000 draws, seed 0) -- in which every resample
of papers is scored on both slices, so the slices' dependence (1,695 papers
contribute pools to both) is kept. It reports the two gains, their
difference and their ratio, and the within-paper contrast on papers present
in both slices.

Expected on test with SPECTER2 at the validation-selected alpha=0.75,
min_count=1, lambda=3.0:

    hard  +0.12382 [+0.11594, +0.13247]   easy  +0.02183 [+0.00974, +0.03324]
    difference +0.10198 [+0.08937, +0.11440]   ratio 5.67 [3.75, 11.86]

Usage:
    .venv/bin/python scripts/hard_easy_contrast.py \\
        --embeddings work/embeddings_test.parquet \\
        --out work/hard_easy_contrast_specter2.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

from locus import config
from locus.core.bootstrap import cluster_ci, paper_mean
from locus.core.probe import reciprocal_rank
from locus.core.rescore import rescore
from locus.eval.sweep import build_pools
from locus.scoring import base_scorer
from locus.scoring.cocitation import CoGraph


def slice_rows(split: str, embeddings: Path | None, texts: Path | None,
               slice_: str, ppmi, lam: float) -> list[tuple[str, float, float]]:
    """(citing_id, base RR, rescored RR) for every pool in the slice."""
    scorer = base_scorer(embeddings, texts)
    rows = []
    for p in build_pools(split, embeddings, slice_, scorer=scorer):
        base = reciprocal_rank(p["base"], p["gold"])
        after = reciprocal_rank(rescore(p["base"], p["anchors"], ppmi, lam), p["gold"])
        rows.append((p["citing_id"], base, after))
    return rows


def per_paper(rows):
    """citing_id -> (mean base RR, mean delta RR) over the paper's pools."""
    acc = collections.defaultdict(lambda: ([], []))
    for g, b, a in rows:
        acc[g][0].append(b)
        acc[g][1].append(a - b)
    return {g: (float(np.mean(b)), float(np.mean(d))) for g, (b, d) in acc.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--embeddings", type=Path)
    ap.add_argument("--texts", type=Path,
                    help="a (key, kind, text) parquet -- score with BM25 instead")
    ap.add_argument("--sweep", type=Path, default=config.WORK_DIR / "sweep_val.json",
                    help="where alpha, min_count and lambda are read from")
    ap.add_argument("--graph", type=Path, default=config.WORK_DIR / "cocitation_train.npz")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=config.WORK_DIR / "hard_easy_contrast_specter2.json")
    args = ap.parse_args(argv)
    if (args.embeddings is None) == (args.texts is None):
        ap.error("exactly one of --embeddings / --texts")

    sel = json.loads(args.sweep.read_text())["best"]
    ppmi = CoGraph.load(args.graph).ppmi(alpha=sel["alpha"], min_count=sel["min_count"])
    lam = sel["lambda"]

    report = {"split": args.split, "selected": sel, "n_boot": args.n_boot,
              "seed": args.seed, "slices": {}}
    papers = {}
    for name, slice_ in (("hard", "headline"), ("easy", "easy")):
        rows = slice_rows(args.split, args.embeddings, args.texts, slice_, ppmi, lam)
        g = [r[0] for r in rows]
        base = [r[1] for r in rows]
        diffs = [r[2] - r[1] for r in rows]
        # The marginal numbers, exactly as `locus eval rank --slice` reports them.
        report["slices"][name] = {
            "pools": len(rows), "papers": len(set(g)),
            "base": paper_mean(base, g),
            "rescored": paper_mean([r[2] for r in rows], g),
            "delta": paper_mean(diffs, g),
            "ci": list(cluster_ci(diffs, g, n_boot=args.n_boot, seed=args.seed)),
        }
        papers[name] = per_paper(rows)
    hard, easy = papers["hard"], papers["easy"]

    # One resample of papers, scored on both slices. A paper absent from a
    # slice contributes nothing to that slice's mean (nanmean), which is what
    # the marginal bootstrap does implicitly by never drawing it.
    keys = sorted(set(hard) | set(easy))
    hd = np.array([hard[k][1] if k in hard else np.nan for k in keys])
    hb = np.array([hard[k][0] if k in hard else np.nan for k in keys])
    ed = np.array([easy[k][1] if k in easy else np.nan for k in keys])
    eb = np.array([easy[k][0] if k in easy else np.nan for k in keys])
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(keys), size=(args.n_boot, len(keys)))
    Hd, Ed = np.nanmean(hd[idx], axis=1), np.nanmean(ed[idx], axis=1)
    Hb, Eb = np.nanmean(hb[idx], axis=1), np.nanmean(eb[idx], axis=1)

    def q(x):
        lo, hi = np.quantile(x, [0.025, 0.975])
        return [float(lo), float(hi)]

    h, e = float(np.nanmean(hd)), float(np.nanmean(ed))
    hbm, ebm = float(np.nanmean(hb)), float(np.nanmean(eb))
    both = [k for k in keys if k in hard and k in easy]
    within = np.array([hard[k][1] - easy[k][1] for k in both])
    widx = rng.integers(0, len(both), size=(args.n_boot, len(both)))
    report["joint"] = {
        "papers": len(keys), "papers_in_both": len(both),
        "difference": {"delta": h - e, "ci": q(Hd - Ed)},
        "ratio": {"value": h / e, "ci": q(Hd / Ed)},
        # Descriptive only: one of several possible normalisations, and it
        # establishes nothing about why the gains differ.
        "headroom_normalised": {
            "hard": {"value": h / (1 - hbm), "ci": q(Hd / (1 - Hb))},
            "easy": {"value": e / (1 - ebm), "ci": q(Ed / (1 - Eb))},
            "ratio": {"value": (h / (1 - hbm)) / (e / (1 - ebm)),
                      "ci": q((Hd / (1 - Hb)) / (Ed / (1 - Eb)))},
        },
        "within_paper": {"delta": float(within.mean()),
                         "ci": q(within[widx].mean(axis=1))},
    }

    for name, s in report["slices"].items():
        print(f"{name:5s} n={s['pools']} papers={s['papers']} base {s['base']:.5f} "
              f"-> {s['rescored']:.5f}  delta {s['delta']:+.5f} "
              f"[{s['ci'][0]:+.5f}, {s['ci'][1]:+.5f}]")
    j = report["joint"]
    print(f"joint paper bootstrap ({args.n_boot} draws, seed {args.seed}, "
          f"{j['papers']} papers, {j['papers_in_both']} in both slices)")
    print(f"  difference {j['difference']['delta']:+.5f} "
          f"[{j['difference']['ci'][0]:+.5f}, {j['difference']['ci'][1]:+.5f}]")
    print(f"  ratio      {j['ratio']['value']:.2f} "
          f"[{j['ratio']['ci'][0]:.2f}, {j['ratio']['ci'][1]:.2f}]")
    hn = j["headroom_normalised"]
    print(f"  headroom-normalised: hard {hn['hard']['value']:.3f} "
          f"easy {hn['easy']['value']:.3f} ratio {hn['ratio']['value']:.2f} "
          f"[{hn['ratio']['ci'][0]:.2f}, {hn['ratio']['ci'][1]:.2f}]")
    print(f"  within-paper hard-easy {j['within_paper']['delta']:+.5f} "
          f"[{j['within_paper']['ci'][0]:+.5f}, {j['within_paper']['ci'][1]:+.5f}]")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
