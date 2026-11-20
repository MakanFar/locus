"""Score the hand labels of the annotation study against the pipeline.

`locus exp annotation-sample` drew the items and wrote a key saying, for each,
what the pipeline decided (anchor by clustering, by markers, both, or not an
anchor), the automatic proximity tier, and the population of the sampling
cell. Two readers labelled the items without seeing any of that:

  G  same citation group      S  same sentence, different group
  W  different sentence       N  not co-located / cannot be determined

This module turns the labels into what docs/annotation_protocol.md section 6
promises:

  agreement     raw agreement and Cohen's kappa between the two readers on
                the four-way label and on the two binary collapses, with the
                confusion matrix; computed on the independent labels, before
                adjudication
  estimates     precision, recall and F1 of the clustering route, the marker
                route and their union against the consensus label, under
                each definition of a site (site = group: only G counts;
                site = sentence: G or S), post-stratified by sampling cell
                and with a paper-level bootstrap interval
  proxy         how often the regular-expression tier agrees with the
                consensus label where it assigns one
  anatomy       false-positive anchors each route adds and true pairs it
                misses, per 100 contexts of the split
  effect        base and rescored MRR on the headline pools whose anchor set
                the labels correct: pools touched by an annotated false
                positive (that anchor removed) and by an annotated false
                negative (that pair added). Needs the embeddings, the graph
                and the sweep selection, like `exp proximity`.

Post-stratification: every item stands for the pairs of its cell, so a cell's
rates are weighted by the cell's population (from the key, originally the
manifest). The interval resamples citing papers with replacement; a cell that
a resample leaves empty keeps its full-sample rates.

Usage:
    .venv/bin/locus exp annotation-score --key reports/annotation_key.csv \\
        --labels reports/annotation_labels.csv --out reports/annotation_score.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import sys
from pathlib import Path

from locus import config

LABELS = ("G", "S", "W", "N")
ROUTES = ("clustering", "marker", "union")
# Which labels count as "same site" under each definition.
DEFINITIONS = {"group": frozenset("G"), "sentence": frozenset("GS")}
LABEL_COLUMNS = ("item_id", "label_A", "label_B", "consensus", "adjudicated")


# ---- labels -----------------------------------------------------------------


def read_key(path: Path) -> list[dict]:
    """The key as rows with typed fields."""
    with open(path, newline="", encoding="utf-8") as f:
        return [
            {
                **r,
                "clustering": r["clustering"] == "True",
                "marker": r["marker"] == "True",
                "cell_population": int(r["cell_population"]),
            }
            for r in csv.DictReader(f)
        ]


def read_labels(path: Path) -> dict[str, dict]:
    """item_id -> {label_A, label_B, consensus, adjudicated}."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for col in ("label_A", "label_B"):
                if r[col] not in LABELS:
                    raise ValueError(f"{path}: {r['item_id']} {col}={r[col]!r}")
            # The pilot file has no consensus: its items were discussed, not
            # adjudicated, and only the readers' agreement on them is used.
            if r["consensus"] and r["consensus"] not in LABELS:
                raise ValueError(f"{path}: {r['item_id']} consensus={r['consensus']!r}")
            out[r["item_id"]] = {**r, "consensus": r["consensus"] or None,
                                 "adjudicated": r["adjudicated"] == "True"}
    return out


def merge_labels(a: dict[str, str], b: dict[str, str],
                 adjudicated: dict[str, str]) -> list[dict]:
    """Consensus rows: the shared label where the readers agree, else the
    adjudicated one. A disagreement without an adjudication is an error, and
    so is an adjudication of an item the readers agreed on."""
    if set(a) != set(b):
        raise ValueError("the two label sets cover different items")
    rows = []
    for item in sorted(a):
        if a[item] == b[item]:
            if item in adjudicated and adjudicated[item] != a[item]:
                raise ValueError(f"{item}: adjudicated but the readers agreed")
            rows.append({"item_id": item, "label_A": a[item], "label_B": b[item],
                         "consensus": a[item], "adjudicated": False})
        else:
            if item not in adjudicated:
                raise ValueError(f"{item}: readers disagree and no consensus is recorded")
            rows.append({"item_id": item, "label_A": a[item], "label_B": b[item],
                         "consensus": adjudicated[item], "adjudicated": True})
    return rows


def write_labels(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in LABEL_COLUMNS})


# ---- agreement --------------------------------------------------------------


def kappa(a: list, b: list) -> tuple[float, float]:
    """Raw agreement and Cohen's kappa for two label sequences."""
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    k = float("nan") if pe == 1.0 else (po - pe) / (1.0 - pe)
    return po, k


def agreement(labels: dict[str, dict]) -> dict:
    items = sorted(labels)
    a = [labels[i]["label_A"] for i in items]
    b = [labels[i]["label_B"] for i in items]
    raw, k4 = kappa(a, b)
    _, kg = kappa([x == "G" for x in a], [x == "G" for x in b])
    _, kgs = kappa([x in "GS" for x in a], [x in "GS" for x in b])
    conf = {x: {y: 0 for y in LABELS} for x in LABELS}
    for x, y in zip(a, b, strict=True):
        conf[x][y] += 1
    return {
        "items": len(items), "raw": raw, "kappa": k4,
        "kappa_g_vs_rest": kg, "kappa_gs_vs_rest": kgs,
        "confusion_A_by_B": conf,
        "disagreements": sum(x != y for x, y in zip(a, b, strict=True)),
        "counts_A": {lab: a.count(lab) for lab in LABELS},
        "counts_B": {lab: b.count(lab) for lab in LABELS},
    }


# ---- post-stratified estimates ----------------------------------------------


def is_anchor(row: dict, route: str) -> bool:
    if route == "clustering":
        return row["clustering"]
    if route == "marker":
        return row["marker"]
    if route == "union":
        return row["clustering"] or row["marker"]
    raise ValueError(route)


def _rates(rows: list[dict], labels: dict[str, str], positive: frozenset,
           route: str) -> dict[str, dict]:
    """Per cell: population and the shares of (anchor, positive) outcomes."""
    cells: dict[str, dict] = {}
    for r in rows:
        c = cells.setdefault(r["cell"], {"pop": r["cell_population"], "n": 0,
                                         "tp": 0, "fp": 0, "fn": 0})
        pos = labels[r["item_id"]] in positive
        anc = is_anchor(r, route)
        c["n"] += 1
        c["tp"] += anc and pos
        c["fp"] += anc and not pos
        c["fn"] += (not anc) and pos
    return cells


def _prf(cells: dict[str, dict], fallback: dict[str, dict] | None = None) -> dict:
    tp = fp = fn = 0.0
    for name, c in (fallback or cells).items():
        got = cells.get(name)
        src = got if got and got["n"] else c
        pop = src["pop"]
        tp += pop * src["tp"] / src["n"]
        fp += pop * src["fp"] / src["n"]
        fn += pop * src["fn"] / src["n"]
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def estimate(rows: list[dict], labels: dict[str, str], definition: str, route: str,
             n_boot: int = 1000, seed: int = config.SEED) -> dict:
    """Post-stratified precision/recall/F1 with a paper-level bootstrap."""
    positive = DEFINITIONS[definition]
    full = _rates(rows, labels, positive, route)
    point = _prf(full)
    by_paper: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_paper[r["citing_id"]].append(r)
    papers = sorted(by_paper)
    rng = random.Random(seed)
    boots: dict[str, list[float]] = {"precision": [], "recall": [], "f1": []}
    for _ in range(n_boot):
        sample = [r for _ in papers for r in by_paper[rng.choice(papers)]]
        est = _prf(_rates(sample, labels, positive, route), fallback=full)
        for k in boots:
            boots[k].append(est[k])

    def ci(xs: list[float]) -> list[float]:
        xs = sorted(x for x in xs if not math.isnan(x))
        if not xs:
            return [float("nan"), float("nan")]
        lo = xs[int(0.025 * (len(xs) - 1))]
        hi = xs[round(0.975 * (len(xs) - 1))]
        return [lo, hi]

    return {**point, "ci": {k: ci(v) for k, v in boots.items()}, "n_boot": n_boot}


def estimates(rows: list[dict], labels: dict[str, str], n_boot: int = 1000,
              seed: int = config.SEED) -> dict:
    return {d: {r: estimate(rows, labels, d, r, n_boot, seed) for r in ROUTES}
            for d in DEFINITIONS}


# ---- the rest of section 6 --------------------------------------------------


def by_stratum(rows: list[dict], labels: dict[str, str]) -> dict:
    out: dict[str, dict] = {}
    for r in rows:
        s = out.setdefault(r["stratum"], {lab: 0 for lab in LABELS})
        s[labels[r["item_id"]]] += 1
    for s in out.values():
        s["n"] = sum(s[lab] for lab in LABELS)
        s["share_N"] = s["N"] / s["n"]
    return out


def proxy_agreement(rows: list[dict], labels: dict[str, str]) -> dict:
    """The regex tier against the consensus label.

    `group`, `sentence` and `beyond` correspond to G, S and W. The
    `not_in_window` tier is not a judgement about the pair but the statement
    that the candidate's marker did not resolve in the target's window; the
    reader then used the second window, so its match rate is reported
    separately and not counted as proxy accuracy.
    """
    conf: dict[str, dict] = {}
    for r in rows:
        conf.setdefault(r["tier"], {lab: 0 for lab in LABELS})[labels[r["item_id"]]] += 1
    expected = {"group": "G", "sentence": "S", "beyond": "W"}
    hit = sum(conf.get(t, {}).get(lab, 0) for t, lab in expected.items())
    assigned = sum(sum(conf.get(t, {}).values()) for t in expected)
    return {
        "confusion_tier_by_label": conf,
        "assigned": assigned, "agree": hit,
        "agreement_where_assigned": hit / assigned if assigned else float("nan"),
    }


def anatomy(est: dict, contexts: int | None) -> dict:
    """False positives added and true pairs missed, per 100 contexts."""
    if not contexts:
        return {"contexts": None}
    out: dict = {"contexts": contexts}
    for d, routes in est.items():
        out[d] = {r: {"fp_per_100_contexts": 100 * v["fp"] / contexts,
                      "fn_per_100_contexts": 100 * v["fn"] / contexts,
                      "tp_per_100_contexts": 100 * v["tp"] / contexts}
                  for r, v in routes.items()}
    return out


def effect_on_result(rows: list[dict], labels: dict[str, str], pools: list[dict],
                     ppmi, lam: float, definition: str, seed: int = config.SEED) -> dict:
    """Rescore the headline pools the labels touch, with their anchors corrected.

    An annotated anchor the consensus does not call a site under `definition`
    is removed from its pool's anchor set; an annotated non-anchor the
    consensus does call a site is added. Three scores per touched pool: the
    base, the rescoring with the pipeline's anchors, and the rescoring with
    the corrected anchors. Reported separately for the pools a false positive
    touches and the pools a false negative touches.
    """
    from locus.eval.rank import _score, paired

    positive = DEFINITIONS[definition]
    by_context = {p["context_id"]: p for p in pools}
    remove: dict[str, set[str]] = collections.defaultdict(set)
    add: dict[str, set[str]] = collections.defaultdict(set)
    for r in rows:
        if r["context_id"] not in by_context:
            continue
        pos = labels[r["item_id"]] in positive
        if is_anchor(r, "union") and not pos:
            remove[r["context_id"]].add(r["candidate"])
        elif not is_anchor(r, "union") and pos:
            add[r["context_id"]].add(r["candidate"])

    def block(touched: dict[str, set[str]]) -> dict:
        sub = [by_context[c] for c in sorted(touched) if touched[c]]
        if not sub:
            return {"pools": 0}
        groups = [p["citing_id"] for p in sub]
        corrected = [frozenset((p["anchors"] - remove.get(p["context_id"], set()))
                               | add.get(p["context_id"], set())) for p in sub]
        base = [_score(p, ppmi, p["anchors"], 0.0) for p in sub]
        pipe = [_score(p, ppmi, p["anchors"], lam) for p in sub]
        fixed = [_score(p, ppmi, a, lam) if a else b
                 for p, a, b in zip(sub, corrected, base, strict=True)]
        vs_base = paired(pipe, base, groups, seed)
        corr_vs_base = paired(fixed, base, groups, seed)
        corr_vs_pipe = paired(fixed, pipe, groups, seed)
        return {
            "pools": len(sub), "papers": len(set(groups)),
            "anchors_removed": sum(len(remove[p["context_id"]]) for p in sub),
            "anchors_added": sum(len(add[p["context_id"]]) for p in sub),
            "pools_left_without_anchors": sum(1 for a in corrected if not a),
            "base": vs_base["b"], "pipeline": vs_base["a"], "corrected": corr_vs_base["a"],
            "delta_pipeline": vs_base["delta"], "delta_pipeline_ci": vs_base["ci"],
            "delta_corrected": corr_vs_base["delta"], "delta_corrected_ci": corr_vs_base["ci"],
            "corrected_minus_pipeline": corr_vs_pipe["delta"],
            "corrected_minus_pipeline_ci": corr_vs_pipe["ci"],
        }

    return {"definition": definition,
            "annotated_contexts_in_headline": sum(
                1 for c in {r["context_id"] for r in rows} if c in by_context),
            "touched_by_false_positives": block(remove),
            "touched_by_false_negatives": block(add)}


# ---- driver -----------------------------------------------------------------


def score(key_rows: list[dict], labels: dict[str, dict], *, contexts: int | None = None,
          pilot: tuple[list[dict], dict[str, dict]] | None = None,
          n_boot: int = 1000, seed: int = config.SEED) -> dict:
    missing = [r["item_id"] for r in key_rows if r["item_id"] not in labels]
    if missing:
        raise ValueError(f"{len(missing)} key items without labels, e.g. {missing[:3]}")
    consensus = {i: v["consensus"] for i, v in labels.items()}
    est = estimates(key_rows, consensus, n_boot, seed)
    report = {
        "protocol_version": "0.3",
        "items": len(key_rows),
        "adjudicated": sum(v["adjudicated"] for v in labels.values()),
        "agreement": agreement(labels),
        "consensus_counts": {lab: sum(c == lab for c in consensus.values()) for lab in LABELS},
        "by_stratum": by_stratum(key_rows, consensus),
        "estimates": est,
        "proxy": proxy_agreement(key_rows, consensus),
        "anatomy": anatomy(est, contexts),
    }
    if pilot is not None:
        prow, plab = pilot
        report["pilot_agreement"] = agreement(plab)
        report["pilot_items"] = len(prow)
    return report


def _print(report: dict) -> None:
    ag = report["agreement"]
    print(f"{report['items']} items, {ag['disagreements']} disagreements adjudicated")
    print(f"agreement raw {ag['raw']:.3f}  kappa {ag['kappa']:.3f}  "
          f"G/rest {ag['kappa_g_vs_rest']:.3f}  G+S/rest {ag['kappa_gs_vs_rest']:.3f}")
    if "pilot_agreement" in report:
        pa = report["pilot_agreement"]
        print(f"pilot ({report['pilot_items']}) raw {pa['raw']:.3f}  kappa {pa['kappa']:.3f}")
    print(f"\n{'site = ':<10}{'route':<12}{'precision':>22}{'recall':>22}{'F1':>8}")
    for d, routes in report["estimates"].items():
        for r, v in routes.items():
            cp, cr = v["ci"]["precision"], v["ci"]["recall"]
            print(f"{d:<10}{r:<12}{v['precision']:>8.3f} [{cp[0]:.3f}, {cp[1]:.3f}]"
                  f"{v['recall']:>8.3f} [{cr[0]:.3f}, {cr[1]:.3f}]{v['f1']:>8.3f}")
    px = report["proxy"]
    print(f"\nproxy tier agrees with consensus on {px['agree']}/{px['assigned']} "
          f"items where it assigns a tier ({px['agreement_where_assigned']:.3f})")
    if report["anatomy"].get("contexts"):
        print(f"\nper 100 contexts ({report['anatomy']['contexts']} in the split)")
        for d in DEFINITIONS:
            for r, v in report["anatomy"][d].items():
                print(f"  {d:<10}{r:<12}FP {v['fp_per_100_contexts']:>6.2f}  "
                      f"FN {v['fn_per_100_contexts']:>6.2f}  TP {v['tp_per_100_contexts']:>6.2f}")
    for d in DEFINITIONS:
        eff = report.get("effect", {}).get(d)
        if not eff:
            continue
        print(f"\neffect on the headline pools, site = {d} "
              f"({eff['annotated_contexts_in_headline']} annotated contexts in the headline set)")
        for name in ("touched_by_false_positives", "touched_by_false_negatives"):
            b = eff[name]
            if not b["pools"]:
                print(f"  {name:<28}no pools")
                continue
            print(f"  {name:<28}{b['pools']} pools  base {b['base']:.5f}  "
                  f"pipeline {b['pipeline']:.5f} ({b['delta_pipeline']:+.5f})  "
                  f"corrected {b['corrected']:.5f} ({b['delta_corrected']:+.5f})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True,
                    help="item_id,label_A,label_B,consensus,adjudicated")
    ap.add_argument("--pilot-key", type=Path, default=None)
    ap.add_argument("--pilot-labels", type=Path, default=None)
    ap.add_argument("--contexts", type=int, default=None,
                    help="contexts in the split, for the per-100-contexts anatomy; "
                         "read from --build-report when given")
    ap.add_argument("--build-report", type=Path, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--embeddings", type=Path, default=None,
                    help="with --graph and --sweep: rescore the touched headline pools")
    ap.add_argument("--sweep", type=Path, default=config.WORK_DIR / "sweep_val.json")
    ap.add_argument("--graph", type=Path, default=config.WORK_DIR / "cocitation_train.npz")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    key_rows = read_key(args.key)
    labels = read_labels(args.labels)
    pilot = None
    if args.pilot_key and args.pilot_labels:
        pilot = (read_key(args.pilot_key), read_labels(args.pilot_labels))
    contexts = args.contexts
    if contexts is None and args.build_report:
        contexts = json.loads(args.build_report.read_text())["contexts"]

    report = score(key_rows, labels, contexts=contexts, pilot=pilot,
                   n_boot=args.n_boot, seed=args.seed)
    report["split"] = args.split

    if args.embeddings:
        from locus.eval.sweep import build_pools
        from locus.scoring.cocitation import CoGraph

        sel = json.loads(args.sweep.read_text())["best"]
        pools = build_pools(args.split, args.embeddings)
        ppmi = CoGraph.load(args.graph).ppmi(alpha=sel["alpha"], min_count=sel["min_count"])
        consensus = {i: v["consensus"] for i, v in labels.items()}
        report["effect"] = {
            d: effect_on_result(key_rows, consensus, pools, ppmi, sel["lambda"], d, args.seed)
            for d in DEFINITIONS
        }
        report["effect"]["selected"] = sel
        report["effect"]["headline_pools"] = len(pools)

    _print(report)
    out = args.out or config.WORK_DIR / "annotation_score.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
