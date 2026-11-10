"""How close is an anchor to the target citation, and what is that worth?

A(l) is inferred, and the inference is generous: the marker route takes every
other citation inside a ~450-character window, which regularly spans three or
four sentences. "Cited at the same location" therefore covers three quite
different things, and this module separates them using nothing but the masked
window:

  group          only punctuation, conjunctions or other markers lie between
                 the anchor's marker and the target's -- one bracket, one run
  sentence       content words lie between them but no sentence ends
  beyond         a sentence ends between them; same window, different sentence
  not_in_window  a clustering-route anchor whose marker is not in this window
                 at all (it is the target of an overlapping window)

Two outputs. First, how the anchors the benchmark uses are distributed over
those tiers. Second, what the reranker gains when it is given the anchors of
ONE tier only, on the pools that have such an anchor -- which is the direct
measurement of whether "location" means the bracket, the sentence or the
window.

The tiers are regular expressions, not judgements. They are the automatic
proxy for the hand labels of the annotation study (same group / same sentence
/ different sentence), stored with each annotation item so their agreement
with human consensus can be reported.

Usage:
    .venv/bin/locus exp proximity --embeddings work/embeddings_test.parquet
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

from locus import config
from locus.core.alignment import _plausible, build_bibliography, markers_with_numbers
from locus.core.types import anchors, anchors_union, gold_sets
from locus.data.corpus import load_index
from locus.eval.rank import _score, paired, pool_level
from locus.eval.sweep import build_pools
from locus.scoring.cocitation import CoGraph

TIERS = ("group", "sentence", "beyond", "not_in_window")
_ORDER = {t: i for i, t in enumerate(TIERS)}

_MARK = re.compile(r"TARGETCIT|OTHERCIT")
# What may sit between two markers of one group. No full stop: a sentence that
# ends right before the second marker is two sentences, not one group.
_FILLER = re.compile(r"(?:TARGETCIT|OTHERCIT|\band\b|\bor\b|&|[\s,;:\[\]()\-\u2013\u2014~])*", re.IGNORECASE)
# A full stop after one of these does not end a sentence. The stop is swapped
# for a one-dot leader so the boundary patterns below cannot see it.
_ABBR = re.compile(
    r"\b(et al|Fig|Figs|Eq|Eqs|Sec|Secs|Ref|Refs|Tab|e\.g|i\.e|cf|vs|resp|approx|No|Dr|Prof)\.",
    re.IGNORECASE,
)
# U+2024 ONE DOT LEADER: stands in for a full stop that does not end a sentence.
_LEADER = chr(0x2024)
_BOUNDARY = re.compile(r"[.!?][\"')\]]*\s+(?=[A-Z\[(])")
_ENDS_SENTENCE = re.compile(r"[.!?][\"')\]]*\s*$")


def tier(between: str) -> str:
    """Tier of two markers given the masked text strictly between them."""
    if _FILLER.fullmatch(between):
        return "group"
    text = _ABBR.sub(lambda m: m.group(0)[:-1] + _LEADER, between)
    if _BOUNDARY.search(text) or _ENDS_SENTENCE.search(_MARK.sub("", text)):
        return "beyond"
    return "sentence"


def tiers_from_markers(
    masked: str,
    marks: list[tuple[str, str]],
    number_to_ref: dict[str, str],
    own_refid: str,
) -> dict[str, str]:
    """refid -> nearest tier, for every resolvable OTHERCIT in one window.

    `marks` is `markers_with_numbers(raw, masked)`: (kind, number) in reading
    order, which is the order `_MARK` finds them in `masked`. A window without
    exactly one TARGETCIT has no reference point and yields nothing. A paper
    cited twice in the window takes its nearest occurrence.
    """
    spans = [(m.start(), m.end()) for m in _MARK.finditer(masked)]
    if len(spans) != len(marks):
        return {}
    targets = [i for i, (kind, _) in enumerate(marks) if kind == "TARGETCIT"]
    if len(targets) != 1:
        return {}
    t = targets[0]
    out: dict[str, str] = {}
    for i, (kind, num) in enumerate(marks):
        if kind != "OTHERCIT" or not _plausible(num):
            continue
        ref = number_to_ref.get(num)
        if ref is None or ref == own_refid:
            continue
        lo, hi = (spans[i][1], spans[t][0]) if i < t else (spans[t][1], spans[i][0])
        got = tier(masked[lo:hi])
        if ref not in out or _ORDER[got] < _ORDER[out[ref]]:
            out[ref] = got
    return out


def anchor_tiers(index: dict) -> tuple[dict[str, dict[str, str]], dict]:
    """context_id -> {anchor: tier} over the union of routes, plus counts.

    The bibliography is rebuilt per citing paper exactly as `data.corpus`
    builds it, so a marker resolves here to the same paper it resolved to when
    the index was frozen.
    """
    per_context: dict[str, dict[str, str]] = {}
    marker_instances: collections.Counter = collections.Counter()
    clustering_instances: collections.Counter = collections.Counter()
    for recs in index.values():
        marks = [markers_with_numbers(r.raw, r.masked) for r in recs]
        bib = build_bibliography(
            [(r.refid, m) for r, m in zip(recs, marks, strict=True) if m is not None]
        )
        golds = gold_sets(recs)
        for rec, m in zip(recs, marks, strict=True):
            seen: dict[str, str] = {}
            if m is not None and rec.alignment_ok:
                seen = tiers_from_markers(rec.masked, m, bib.number_to_ref, rec.refid)
            for a in rec.marker_anchors:
                marker_instances[seen.get(a, "not_in_window")] += 1
            for a in anchors(rec, golds):
                clustering_instances[seen.get(a, "not_in_window")] += 1
            per_context[rec.context_id] = {
                a: seen.get(a, "not_in_window") for a in anchors_union(rec, golds)
            }
    return per_context, {
        "marker_route_instances": dict(marker_instances),
        "clustering_route_instances": dict(clustering_instances),
    }


def gain_with(pools: list[dict], anchor_sets: dict[str, set[str]], ppmi, lam: float,
              seed: int, agg: str = "mean") -> dict:
    """Base vs rescored on the pools whose restricted anchor set is non-empty.

    A pool with no anchor in the set is left out rather than scored as a zero
    gain: the question is what an anchor of this kind is worth where one
    exists, and averaging in the pools without one answers a different
    question (how common the kind is), which the counts already report.
    """
    sub = [(p, frozenset(anchor_sets.get(p["context_id"], ()))) for p in pools]
    sub = [(p, a) for p, a in sub if a]
    if not sub:
        return {"n": 0, "papers": 0, "a": None, "b": None, "delta": None, "ci": [None, None],
                "pool_a": None, "pool_b": None, "pool_delta": None, "mean_anchors": None}
    groups = [p["citing_id"] for p, _ in sub]
    base = [_score(p, ppmi, a, 0.0, agg=agg) for p, a in sub]
    resc = [_score(p, ppmi, a, lam, agg=agg) for p, a in sub]
    row = paired(resc, base, groups, seed)
    pl = pool_level(resc, base, groups, seed)
    row["pool_a"], row["pool_b"], row["pool_delta"] = pl["a"], pl["b"], pl["delta"]
    row["pool_ci"] = pl["ci"]
    row["mean_anchors"] = sum(len(a) for _, a in sub) / len(sub)
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test")
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--sweep", type=Path, default=config.WORK_DIR / "sweep_val.json")
    ap.add_argument("--graph", type=Path, default=config.WORK_DIR / "cocitation_train.npz")
    ap.add_argument("--tag", default="specter2")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    sel = json.loads(args.sweep.read_text())["best"]
    lam = sel["lambda"]
    index = load_index(config.WORK_DIR / f"index_{args.split}.pkl")
    tiers_of, counts = anchor_tiers(index)

    closest = collections.Counter(
        min(t.values(), key=_ORDER.get) for t in tiers_of.values() if t
    )
    pools = build_pools(args.split, args.embeddings)
    ppmi = CoGraph.load(args.graph).ppmi(alpha=sel["alpha"], min_count=sel["min_count"])
    headline_closest = collections.Counter(
        min(tiers_of[p["context_id"]].values(), key=_ORDER.get) for p in pools
    )

    def restricted(keep: tuple[str, ...]) -> dict[str, set[str]]:
        return {c: {a for a, t in ts.items() if t in keep} for c, ts in tiers_of.items()}

    one_tier = {t: gain_with(pools, restricted((t,)), ppmi, lam, config.SEED) for t in TIERS}
    cumulative = {
        "+".join(TIERS[:k]): gain_with(pools, restricted(TIERS[:k]), ppmi, lam, config.SEED)
        for k in range(1, len(TIERS) + 1)
    }

    # Dilution, on a fixed set of pools: those with a same-group anchor. Only
    # the anchor set changes down the rows, so a falling delta is the cost of
    # adding looser anchors and not a change in which pools are averaged.
    has_group = [p for p in pools if "group" in tiers_of[p["context_id"]].values()]
    fixed_cohort = {
        "+".join(TIERS[:k]): gain_with(has_group, restricted(TIERS[:k]), ppmi, lam, config.SEED)
        for k in range(1, len(TIERS) + 1)
    }

    def shares(counter) -> dict:
        total = sum(counter.values())
        return {t: {"n": counter.get(t, 0), "share": counter.get(t, 0) / total} for t in TIERS}

    report = {
        "tag": args.tag, "split": args.split, "selected": sel, "averaging": "paper",
        "pools": len(pools),
        "marker_route_instances": shares(counts["marker_route_instances"]),
        "clustering_route_instances": shares(counts["clustering_route_instances"]),
        "anchored_contexts_by_closest_tier": shares(closest),
        "headline_pools_by_closest_tier": shares(headline_closest),
        "gain_one_tier_only": one_tier,
        "gain_cumulative": cumulative,
        "gain_fixed_cohort_with_group_anchor": fixed_cohort,
    }

    print(f"{len(pools)} headline pools | lambda={lam}")
    for name in ("marker_route_instances", "clustering_route_instances",
                 "headline_pools_by_closest_tier"):
        print(f"\n{name}")
        for t in TIERS:
            row = report[name][t]
            print(f"  {t:<14}{row['n']:>8}  {row['share']:.1%}")
    for title, block in (("anchors of ONE tier only", one_tier),
                         ("cumulative, nearest first", cumulative),
                         ("same pools (have a group anchor)", fixed_cohort)):
        print(f"\n{title:<34}{'pools':>7}{'|A|':>6}{'base':>9}{'+resc':>9}{'delta':>10}  95% CI")
        for name, row in block.items():
            if not row["n"]:
                print(f"  {name:<32}{0:>7}")
                continue
            print(f"  {name:<32}{row['n']:>7}{row['mean_anchors']:>6.2f}{row['b']:>9.5f}"
                  f"{row['a']:>9.5f}{row['delta']:>+10.5f}  "
                  f"[{row['ci'][0]:+.5f}, {row['ci'][1]:+.5f}]")

    out = args.out or config.WORK_DIR / f"proximity_{args.tag}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
