"""Draw the sample two annotators label to check the inferred locations.

Locations and anchors are inferred from text windows, with no ground truth.
This module produces what a hand-label study needs: a stratified sample of
(target citation, candidate citation) pairs from the test split, a sheet per
annotator that shows the text and nothing about what the pipeline decided,
and a key that holds everything the sheets hide.

The unit is an ordered pair inside one citing paper: a context (the target
citation and its window) and another paper the same citing paper cites. Every
such pair falls in exactly one stratum, chosen so the sample covers the
pipeline's likely MERGES and its likely SPLITS rather than only the pairs
near the clustering threshold:

  both_routes                anchor by clustering and by markers
  marker_only                marker resolves in the window; different cluster
  clustering_only_aligned    same cluster, markers parsed, but the candidate's
                             marker does not resolve in this window
  clustering_only_unaligned  same cluster, and this window failed alignment,
                             so clustering is the only evidence there is
  transitive                 same cluster only through a third window: the
                             two windows themselves are below the threshold.
                             Outranks the strata above -- this is where
                             union-find can over-merge
  near_miss                  not an anchor; windows overlap, Jaccard in
                             [0.2, theta): where the pipeline may have split
  weak_overlap               not an anchor; Jaccard in (0, 0.2)
  disjoint                   not an anchor; windows share nothing (a sanity
                             stratum: every label should be "not co-located")

Within the two strata where the candidate is visible in the target's window
the draw is split by the automatic proximity tier (same group / same
sentence / other sentence), so likely-G and likely-W pairs are both present.
Each (stratum, tier) cell records its population, which is what turns the
labels into post-stratified estimates for the whole split.

At most two items come from any one citing paper. A pilot sample is drawn
first, with its own seed, and the main sample excludes it.

The sheets contain corpus text; they are written under the work directory and
are not for redistribution unless the upstream licence allows it. The key and
the manifest contain identifiers and numbers only.

Usage:
    .venv/bin/locus exp annotation-sample --out work/annotation
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import sys
from pathlib import Path

from locus import config
from locus.core.alignment import _plausible, build_bibliography, markers_with_numbers
from locus.core.locations import _jaccard, shingles
from locus.core.types import ContextRec, anchors, gold_sets
from locus.data.corpus import load_index
from locus.experiments.proximity import _MARK, tiers_from_markers

PROTOCOL_VERSION = "0.3"

STRATA = (
    "both_routes", "marker_only", "clustering_only_aligned",
    "clustering_only_unaligned", "transitive", "near_miss", "weak_overlap",
    "disjoint",
)
QUOTAS = {
    "both_routes": 60, "marker_only": 80, "clustering_only_aligned": 40,
    "clustering_only_unaligned": 60, "transitive": 60, "near_miss": 60,
    "weak_overlap": 20, "disjoint": 20,
}
PILOT_QUOTAS = {
    "both_routes": 6, "marker_only": 6, "clustering_only_aligned": 4,
    "clustering_only_unaligned": 5, "transitive": 5, "near_miss": 6,
    "weak_overlap": 4, "disjoint": 4,
}
# Strata whose candidate is visible in the target's window are split by the
# automatic proximity tier, as shares of the stratum's quota. Where the routes
# agree the split is even. A marker-only anchor inside the target's own
# citation group is an anomaly (a few hundred pairs in the split), so it is
# sampled but does not take a third of its stratum.
TIER_SHARES = {
    "both_routes": {"group": 1 / 3, "sentence": 1 / 3, "beyond": 1 / 3},
    "marker_only": {"group": 1 / 8, "sentence": 7 / 16, "beyond": 7 / 16},
}
BALANCED = tuple(TIER_SHARES)
NEAR_MISS_FLOOR = 0.2
PER_PAPER = 2
# Candidates kept per cell while scanning. The disjoint cell alone holds over
# a million pairs; a uniform reservoir keeps the draw uniform without holding
# them all, and the cell's true population is counted separately.
RESERVOIR = 20_000

TARGET, CANDIDATE, OTHER = "⟦T⟧", "⟦C⟧", "[cit]"

SHEET_COLUMNS = ("item_id", "text", "text_2", "label", "uncertain", "comment")
KEY_COLUMNS = (
    "item_id", "stratum", "cell", "tier", "jaccard", "clustering", "marker",
    "direct", "aligned", "citing_id", "context_id", "target", "candidate",
    "candidate_context", "cell_population", "cell_drawn",
)


def stratum_of(*, clustering: bool, marker: bool, direct: bool, aligned: bool,
               jaccard: float, theta: float = config.THETA) -> str:
    """The one stratum an ordered (context, candidate) pair belongs to.

    `direct` says the two windows themselves reach the threshold; a clustering
    anchor without it was joined through a third window.
    """
    if clustering and not direct:
        return "transitive"
    if clustering and marker:
        return "both_routes"
    if marker:
        return "marker_only"
    if clustering:
        return "clustering_only_aligned" if aligned else "clustering_only_unaligned"
    if jaccard >= min(NEAR_MISS_FLOOR, theta):
        return "near_miss"
    return "weak_overlap" if jaccard > 0.0 else "disjoint"


def render(masked: str, marks: list[tuple[str, str]] | None,
           number_to_ref: dict[str, str], candidate: str,
           own: str = TARGET, other: str = CANDIDATE) -> str:
    """A masked window as the annotator sees it.

    The window's own citation becomes `own`, every marker that resolves to
    `candidate` becomes `other`, and every remaining marker becomes "[cit]".
    Without marker numbers (`marks` is None, or does not line up with the
    text) nothing can be resolved, so only the window's own citation is
    marked. Reference numbers the corpus left unmasked stay as they are.
    """
    spans = list(_MARK.finditer(masked))
    resolvable = marks is not None and len(marks) == len(spans)
    out, pos = [], 0
    for i, m in enumerate(spans):
        out.append(masked[pos:m.start()])
        if m.group(0) == "TARGETCIT":
            out.append(own)
        elif (resolvable and _plausible(marks[i][1])
              and number_to_ref.get(marks[i][1]) == candidate):
            out.append(other)
        else:
            out.append(OTHER)
        pos = m.end()
    out.append(masked[pos:])
    return "".join(out)


def pairs_in_paper(recs: list[ContextRec], theta: float = config.THETA,
                   shingle_n: int = config.SHINGLE_N, with_text: bool = True) -> list[dict]:
    """Every ordered (context, other cited paper) pair of one citing paper.

    One row per pair even when the candidate is cited several times: the
    candidate's window shown, and the Jaccard recorded, are those of its
    occurrence nearest the target's window.
    """
    recs = sorted(recs, key=lambda r: r.context_id)
    marks = [markers_with_numbers(r.raw, r.masked) for r in recs]
    bib = build_bibliography(
        [(r.refid, m) for r, m in zip(recs, marks, strict=True) if m is not None])
    golds = gold_sets(recs)
    sh = [shingles(r.raw, shingle_n) for r in recs]
    contexts_of: dict[str, list[int]] = collections.defaultdict(list)
    for j, r in enumerate(recs):
        contexts_of[r.refid].append(j)

    rows = []
    for i, rec in enumerate(recs):
        usable = marks[i] if rec.alignment_ok else None
        seen = (tiers_from_markers(rec.masked, usable, bib.number_to_ref, rec.refid)
                if usable is not None else {})
        cluster_anchors = anchors(rec, golds)
        for cand in sorted(contexts_of):
            if cand == rec.refid:
                continue
            sims = [(_jaccard(sh[i], sh[j]), j) for j in contexts_of[cand]]
            best_j = max(sims, key=lambda t: (t[0], -t[1]))[1]
            jac = max(s for s, _ in sims)
            in_cluster = cand in cluster_anchors
            direct = in_cluster and any(
                s >= theta for s, j in sims if recs[j].location == rec.location)
            in_marker = cand in rec.marker_anchors
            row = {
                "citing_id": rec.citing_id, "context_id": rec.context_id,
                "target": rec.refid, "candidate": cand,
                "candidate_context": recs[best_j].context_id,
                "clustering": in_cluster, "marker": in_marker, "direct": direct,
                "aligned": rec.alignment_ok, "jaccard": jac,
                "tier": seen.get(cand, "not_in_window"),
                "stratum": stratum_of(clustering=in_cluster, marker=in_marker, direct=direct,
                                      aligned=rec.alignment_ok, jaccard=jac, theta=theta),
            }
            if with_text:
                text = render(rec.masked, usable, bib.number_to_ref, cand)
                row["text"] = text
                # The candidate's own window, with the roles swapped, when it
                # could not be marked in the target's.
                row["text_2"] = "" if CANDIDATE in text else render(
                    recs[best_j].masked,
                    marks[best_j] if recs[best_j].alignment_ok else None,
                    bib.number_to_ref, rec.refid, own=CANDIDATE, other=TARGET)
            rows.append(row)
    return rows


def cell_of(row: dict) -> tuple[str, str]:
    """The sampling cell: the stratum, split by tier where the draw is balanced."""
    if row["stratum"] in BALANCED:
        return row["stratum"], row["tier"]
    return row["stratum"], "all"


def cell_quotas(quotas: dict[str, int]) -> dict[tuple[str, str], int]:
    """Per-cell quotas: a tiered stratum is split by TIER_SHARES.

    Shares are rounded down and the remainder goes to the tiers in order, so
    the cells of a stratum always sum to its quota.
    """
    out = {}
    for stratum, n in quotas.items():
        if stratum not in TIER_SHARES:
            out[stratum, "all"] = n
            continue
        shares = TIER_SHARES[stratum]
        cells = {tier: int(n * share) for tier, share in shares.items()}
        for tier in list(shares)[: n - sum(cells.values())]:
            cells[tier] += 1
        for tier, k in cells.items():
            out[stratum, tier] = k
    return out


def draw(cells: dict[tuple[str, str], list[dict]], quotas: dict[tuple[str, str], int],
         seed: int, exclude: set[tuple[str, str]] | None = None,
         per_paper: int = PER_PAPER,
         populations: dict[tuple[str, str], int] | None = None) -> dict:
    """Up to `quota` items per cell, at most `per_paper` from one citing paper.

    A cell that cannot be filled is reported short, never padded from another
    cell: the population each item stands for is what the estimates rest on.
    `populations` gives a cell's true size when `cells` holds only a uniform
    subsample of it.
    """
    exclude = exclude or set()
    rng = random.Random(seed)
    taken: collections.Counter = collections.Counter()
    items, report = [], {}
    for cell in sorted(quotas):
        pool = sorted(cells.get(cell, []), key=lambda r: (r["context_id"], r["candidate"]))
        rng.shuffle(pool)
        got = 0
        for row in pool:
            if got == quotas[cell]:
                break
            if (row["context_id"], row["candidate"]) in exclude:
                continue
            if taken[row["citing_id"]] >= per_paper:
                continue
            taken[row["citing_id"]] += 1
            items.append({**row, "cell": "/".join(cell)})
            got += 1
        size = (populations or {}).get(cell, len(pool))
        report[cell] = {"population": size, "quota": quotas[cell], "drawn": got}
    return {"items": items, "cells": report}


def order_for(item_ids: list[str], annotator: str, seed: int) -> list[str]:
    """The order one annotator sees the items in; differs between annotators."""
    out = sorted(item_ids)
    random.Random(f"{seed}:{annotator}").shuffle(out)
    return out


def write_sheet(path: Path, items: list[dict], order: list[str]) -> None:
    """The annotator's sheet: the text and three empty columns, nothing else."""
    by_id = {it["item_id"]: it for it in items}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        for item_id in order:
            it = by_id[item_id]
            w.writerow({"item_id": item_id, "text": it["text"], "text_2": it["text_2"],
                        "label": "", "uncertain": "", "comment": ""})


def write_key(path: Path, items: list[dict], cells: dict) -> None:
    """What the sheets hide: the pipeline's decision and the sampling weights."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=KEY_COLUMNS)
        w.writeheader()
        for it in sorted(items, key=lambda r: r["item_id"]):
            cell = tuple(it["cell"].split("/"))
            w.writerow({**{k: it[k] for k in KEY_COLUMNS if k in it},
                        "jaccard": f"{it['jaccard']:.4f}",
                        "cell_population": cells[cell]["population"],
                        "cell_drawn": cells[cell]["drawn"]})


def _scan(index: dict, theta: float, seed: int) -> tuple[dict, dict]:
    """Uniform reservoirs of light rows per cell, and each cell's population."""
    rng = random.Random(seed)
    reservoirs: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    population: collections.Counter = collections.Counter()
    for citing in sorted(index):
        for row in pairs_in_paper(index[citing], theta, with_text=False):
            cell = cell_of(row)
            population[cell] += 1
            bucket = reservoirs[cell]
            if len(bucket) < RESERVOIR:
                bucket.append(row)
            else:
                k = rng.randrange(population[cell])
                if k < RESERVOIR:
                    bucket[k] = row
    return reservoirs, dict(population)


def _with_text(items: list[dict], index: dict, theta: float) -> list[dict]:
    """Attach the rendered windows to the drawn items only."""
    wanted = collections.defaultdict(set)
    for it in items:
        wanted[it["citing_id"]].add((it["context_id"], it["candidate"]))
    text = {}
    for citing, pairs in wanted.items():
        for row in pairs_in_paper(index[citing], theta, with_text=True):
            key = (row["context_id"], row["candidate"])
            if key in pairs:
                text[key] = (row["text"], row["text_2"])
    return [{**it, "text": text[it["context_id"], it["candidate"]][0],
             "text_2": text[it["context_id"], it["candidate"]][1]} for it in items]


def _label(items: list[dict], prefix: str, seed: int) -> list[dict]:
    """Opaque ids, assigned in a shuffled order so an id says nothing."""
    order = sorted(items, key=lambda r: (r["context_id"], r["candidate"]))
    random.Random(f"{seed}:{prefix}").shuffle(order)
    width = len(str(len(order)))
    return [{**it, "item_id": f"{prefix}{n:0{width}d}"} for n, it in enumerate(order, 1)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", type=Path, default=config.WORK_DIR / "annotation")
    args = ap.parse_args(argv)

    index = load_index(config.WORK_DIR / f"index_{args.split}.pkl")
    reservoirs, population = _scan(index, config.THETA, args.seed)

    pilot = draw(reservoirs, cell_quotas(PILOT_QUOTAS), seed=args.seed + 1,
                 populations=population)
    used = {(it["context_id"], it["candidate"]) for it in pilot["items"]}
    main_draw = draw(reservoirs, cell_quotas(QUOTAS), seed=args.seed, exclude=used,
                     populations=population)

    args.out.mkdir(parents=True, exist_ok=True)
    items = _label(_with_text(main_draw["items"], index, config.THETA), "L", args.seed)
    pilot_items = _label(_with_text(pilot["items"], index, config.THETA), "P", args.seed)
    ids = [it["item_id"] for it in items]
    for annotator in ("A", "B"):
        write_sheet(args.out / f"sheet_{annotator}.csv", items,
                    order_for(ids, annotator, args.seed))
    write_key(args.out / "key.csv", items, main_draw["cells"])
    write_sheet(args.out / "pilot_sheet.csv", pilot_items,
                order_for([it["item_id"] for it in pilot_items], "pilot", args.seed))
    write_key(args.out / "pilot_key.csv", pilot_items, pilot["cells"])

    quotas = cell_quotas(QUOTAS)
    manifest = {
        "protocol_version": PROTOCOL_VERSION, "split": args.split, "seed": args.seed,
        "theta": config.THETA, "per_paper": PER_PAPER,
        "items": len(items), "pilot_items": len(pilot_items),
        "pairs_in_split": sum(population.values()),
        "cells": {"/".join(c): v for c, v in sorted(main_draw["cells"].items())},
        "pilot_cells": {"/".join(c): v for c, v in sorted(pilot["cells"].items())},
        # Pairs no cell quota covers; reported so the estimates say what they
        # leave out.
        "unsampled_cells": {"/".join(c): n for c, n in sorted(population.items())
                            if c not in quotas},
    }
    with open(args.out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"{manifest['pairs_in_split']} ordered pairs in {args.split}")
    print(f"{'cell':<36}{'population':>12}{'quota':>7}{'drawn':>7}")
    for cell, v in manifest["cells"].items():
        flag = "" if v["drawn"] == v["quota"] else "   SHORT"
        print(f"{cell:<36}{v['population']:>12}{v['quota']:>7}{v['drawn']:>7}{flag}")
    for cell, n in manifest["unsampled_cells"].items():
        print(f"{cell:<36}{n:>12}{'-':>7}{'-':>7}   not sampled")
    print(f"\n{len(items)} items and a pilot of {len(pilot_items)} written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
