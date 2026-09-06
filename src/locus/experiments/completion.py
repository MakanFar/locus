"""Sequential citation-set completion.

At a location with true set G(l), |G(l)| >= 2, ONE true citation is revealed
as a seed -- the same seed, in the same seeded order, for every arm -- and the
system then proposes the rest one at a time from the location's oracle
prefetch. Recall@K counts the seed plus the first K proposals:

    Recall@K = |({seed} + predicted[:K]) intersect G(l)| / |G(l)|

Every arm is handed exactly one true citation and nothing else, so the
comparison is of what each does with the same information. Only the anchor
set the rescorer conditions on differs:

  semantic          no anchors. Cosine alone, so its ranking never moves and
                    revealing a citation cannot help it. The baseline.
  hybrid_retrieved  anchors are the seed plus the system's OWN proposals,
                    mistakes included. The realistic setting, and where error
                    compounds.
  hybrid_oracle     anchors are the seed plus only those proposals that were
                    actually correct. Identical to retrieved except that its
                    anchors never contain an error, so the oracle-retrieved
                    gap *is* the cost of compounding error. It does not see
                    unfound members of G(l) -- an oracle defined that way
                    would simply be handed the answer.
  hybrid_random     same anchor count as retrieved, drawn from the graph.
  hybrid_degree     same count, matched on marginal degree.

Revelation order is seeded and averaged over several seeds:
ground-truth marker order correlates with position in the sentence, so using
it would confound order with location.

Association sums are accumulated incrementally -- one anchor added per step --
rather than recomputed over the whole anchor set, through the measure's own
`prepare`/`add_to` pair. For PPMI that is a CSR row scattered into the
candidates sharing a column, about seven nonzeros, so a step costs O(nnz)
rather than O(candidates); for node2vec every candidate has a vector and the
step is one matrix-vector product. Unlike `aggregate` this accumulates with
+=, which is safe here only because the anchor sequence is itself
deterministic.

Usage:
    .venv/bin/python -m locus.experiments.completion \
        --embeddings work/embeddings_expb_test.parquet --sweep work/sweep_val.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from locus import config
from locus.core.bootstrap import cluster_ci, paper_mean
from locus.eval.controls import degree_bins
from locus.scoring.cocitation import CoGraph
from locus.scoring.dense import DenseScorer
from locus.scoring.mmapstore import MmapScorer
from locus.scoring.node2vec import ASSOC_CHOICES, association

KS = (1, 2, 3, 5, 10, 20)
N_BOOT = 1000   # paper-level bootstrap resamples; section 4.2 says 1000 for
                # every interval in the paper, so this is not a tuning knob


class Recalls(dict):
    """Recall@K keyed by K, with the seed-excluded view carried alongside.

    Indexing gives the SEED-INCLUSIVE recall, unchanged:

        R@K = |({seed} + predicted[:K]) & G(l)| / |G(l)|

    which is what every published completion number is. It hands each arm
    1/|G(l)| for free, and with 56.6% of test locations at |G(l)| = 2 that
    free half dominates: it compresses the ratio between arms toward 1 and
    makes the seed-inclusive numbers useless for the question "given one
    known member, how much more of the REST does conditioning recover".

    `.remaining` answers that question instead, over G(l) minus the seed:

        R_rem@K = |predicted[:K] & (G(l) - {seed})| / (|G(l)| - 1)

    It is `None` when there is no revealed seed (self-seeded mode), because
    "the remaining members" is not defined there -- returning the inclusive
    numbers under a second name would silently make the two modes look
    comparable, which is the same trap `reveal_seed=False` already avoids
    for recall credit.
    """

    __slots__ = ("remaining",)

    def __init__(self, remaining: dict[int, float] | None = None) -> None:
        super().__init__()
        self.remaining = remaining
ARMS = ("semantic", "hybrid_oracle", "hybrid_retrieved", "hybrid_random",
        "hybrid_degree")


def revelation_order(gold: list[str], seed: int, key: str) -> list[str]:
    rng = random.Random(f"{seed}:{key}")
    out = sorted(gold)
    rng.shuffle(out)
    return out


def by_set_size(values, groups, sizes, cap: int = 5) -> dict[int, dict]:
    """Group per-row values by how many citations the location carries.

    `cap` collapses the tail into one bucket: 369 of 22,274 test locations
    carry ten or more citations and a bucket per size would report noise.

    `paper_mean` is used inside each bucket for the same reason it is used
    overall, which means the buckets do NOT recombine into the headline
    number -- a citing paper contributes locations to several of them and is
    weighted once in each.
    """
    grouped: dict[int, tuple[list[float], list[str]]] = {}
    for value, group, size in zip(values, groups, sizes, strict=True):
        vals, keys = grouped.setdefault(min(size, cap), ([], []))
        vals.append(value)
        keys.append(group)
    return {size: {"mean": paper_mean(*grouped[size]), "rows": len(grouped[size][0])}
            for size in sorted(grouped)}


def base_scores(scorer, loc) -> np.ndarray:
    """Cosine over a location's candidates. Computed ONCE per location.

    This used to be inlined as
        [scorer.pool_scores(ctx, cands)[c] for c in cands]
    which re-ran the whole 2,000-candidate gather once per candidate -- 3.9 s
    a call, and quadratic in the pool. The arms all share the same base, so it
    is hoisted out of the arm loop as well.
    """
    cands = loc["candidates"]
    scores = scorer.pool_scores(loc["contexts"][0], cands)
    return np.asarray([scores[c] for c in cands], dtype=np.float32)


STANDARDIZE_CHOICES = ("zscore", "none")


def standardize_base(base: np.ndarray, how: str) -> np.ndarray:
    """Put the base on the scale Eq. 1's lambda was selected on.

    `zscore` standardises over the location's candidate list -- the same
    convention `core.rescore.zscore_pool` applies to a 10-candidate pool, and
    the one the val sweep that chose lambda ran under. A zero-variance base
    (every candidate tied on cosine) becomes all zeros rather than NaN, again
    as in `zscore_pool`. `none` is the raw cosine, kept so the numbers
    published before 2026-09-05 can be regenerated; it is not Eq. 1.
    """
    if how == "none":
        return base
    if how != "zscore":
        raise ValueError(
            f"unknown standardize {how!r}; expected one of {STANDARDIZE_CHOICES}"
        )
    vals = base.astype(np.float64)
    mu = vals.mean()
    sd = vals.std()  # population sd, matching zscore_pool
    if sd == 0.0:
        return np.zeros_like(base, dtype=np.float32)
    return ((vals - mu) / sd).astype(np.float32)


def run_location(loc, scorer, ppmi, graph, bins, lam, seed, arm, base=None,
                 reveal_seed: bool = True, trace: list | None = None,
                 prep=None, standardize: str = "zscore"):
    """One location, one arm.

    `reveal_seed` picks between the two information states the paper reports.

    True (default) hands every arm the same single TRUE citation and blocks it
    from being proposed. The arms then differ only in what they condition on
    afterwards, so oracle-vs-retrieved isolates the cost of compounding error.

    False is the strictly harder setting: A_0 = {} and the system must create
    its own first anchor from the base ranking alone, which it can get wrong.
    The seed contributes no recall credit in this mode -- counting it would
    hand every arm 1/|G| for free and make the two modes look comparable when
    they are not.

    `standardize` is how the base is scaled before lambda * PPMI is added --
    see `standardize_base`. The default is Eq. 1's z-score. `base`, when
    passed, is the RAW cosine vector; standardisation happens here so every
    arm sees the same transform of the same shared base.
    """
    gold = set(loc["gold"])
    cands = loc["candidates"]
    key = f"{loc['citing_id']}:{loc['location']}"
    order = revelation_order(loc["gold"], seed, key)
    seed_paper = order[0] if reveal_seed else None

    base = base_scores(scorer, loc) if base is None else base
    base = standardize_base(base, standardize)
    # The association measure owns the per-location layout: a column index for
    # the sparse counts, a dense matrix for node2vec. Branching on the measure
    # here instead would put the one thing the arms are supposed to differ in
    # inside the loop that is supposed to be identical between them.
    #
    # Hoisted out of the arm loop for the same reason `base` is: it depends
    # only on the candidate list, and node2vec's layout is a 2,150 x 128
    # matrix that would otherwise be rebuilt five times per location.
    prep = ppmi.prepare(cands) if prep is None else prep
    idx_of = {c: i for i, c in enumerate(cands)}

    sums = np.zeros(len(cands), dtype=np.float64)
    n_anchors = 0
    n_correct = 0
    blocked = np.zeros(len(cands), dtype=bool)
    if seed_paper is not None and seed_paper in idx_of:
        blocked[idx_of[seed_paper]] = True

    def add_anchor(paper: str) -> None:
        nonlocal n_anchors, n_correct
        n_anchors += 1
        n_correct += paper in gold
        ppmi.add_to(sums, paper, prep)

    rng = random.Random(f"{seed}:{arm}:{key}")
    of_paper, members, keys = bins

    def _substitute(paper: str) -> str:
        """The random / degree-matched stand-in for a real anchor."""
        if arm == "hybrid_random":
            return graph.vocab[rng.randrange(len(graph.vocab))]
        b = of_paper.get(paper)
        bucket = members[0] if b is None else members[
            min(range(len(keys)), key=lambda i: abs(keys[i] - b))]
        return bucket[rng.randrange(len(bucket))]

    hybrid = arm != "semantic"
    if hybrid and seed_paper is not None:
        # The seed anchor is substituted too in the random and degree arms.
        # Seeding every arm with the TRUE revealed citation and randomising
        # only what follows makes them controls on anchors-after-the-first,
        # not on anchor quality -- and since the seed carries most of the
        # signal, it collapses the spectrum: every arm scored an identical
        # R@1 and random came within 0.008 of retrieved at R@20.
        add_anchor(seed_paper if arm in ("hybrid_oracle", "hybrid_retrieved")
                   else _substitute(seed_paper))
    predicted: list[str] = []
    # |G(l)| = 1 cannot have a "remaining" set; the task file excludes those,
    # but a caller passing one in should get None rather than a ZeroDivision.
    recalls = Recalls({} if seed_paper is not None and len(gold) > 1 else None)
    for step in range(1, max(KS) + 1):
        held, held_ok = n_anchors, n_correct
        scores = base + (lam * (sums / n_anchors)).astype(np.float32) \
            if hybrid and n_anchors else base
        scores = np.where(blocked, -np.inf, scores)
        pick = int(np.argmax(scores))
        chosen = cands[pick]
        blocked[pick] = True
        predicted.append(chosen)
        if trace is not None:
            trace.append({"step": step, "pick": chosen,
                          "correct": chosen in gold,
                          "n_anchors": held, "n_correct": held_ok})
        if step in KS:
            found = set(predicted[:step])
            # The seed is blocked, so `predicted` can never contain it and
            # this intersection is ALREADY over G(l) minus the seed -- the
            # seed-excluded recall needs no separate bookkeeping, only a
            # different denominator. Computed before the seed is added to
            # `found`, which is why the order of these three lines matters.
            if recalls.remaining is not None:
                recalls.remaining[step] = len(found & gold) / (len(gold) - 1)
            if seed_paper is not None:
                found.add(seed_paper)
            recalls[step] = len(found & gold) / len(gold)

        if not hybrid:
            continue
        if arm == "hybrid_retrieved":
            add_anchor(chosen)
        elif arm == "hybrid_oracle":
            if chosen in gold:          # only correct anchors, never an error
                add_anchor(chosen)
        else:
            add_anchor(_substitute(chosen))
    return recalls


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test")
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--sweep", type=Path, default=config.WORK_DIR / "sweep_val.json")
    ap.add_argument("--graph", type=Path,
                    default=config.WORK_DIR / "cocitation_train.npz")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--mode", choices=("revealed", "selfseed"),
                    default="revealed",
                    help="revealed: one true citation is handed to every arm. "
                         "selfseed: A_0 = {} and the system must retrieve its "
                         "own first anchor, which it can get wrong.")
    ap.add_argument("--standardize", choices=STANDARDIZE_CHOICES,
                    default="zscore",
                    help="how the base is scaled before lambda * PPMI is "
                         "added. zscore (default) is Eq. 1 and the scale "
                         "lambda was selected on; none is the raw cosine the "
                         "pre-2026-09-05 numbers were produced with.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--task", type=Path, default=None,
                    help="task jsonl; defaults to the oracle-prefetch file. "
                         "Point at expb_task_<split>_corpuswide.jsonl for the "
                         "end-to-end setting, where the gold is NOT guaranteed "
                         "to be a candidate.")
    ap.add_argument("--assoc", choices=ASSOC_CHOICES, default="ppmi",
                    help="association measure; must match the --sweep used, "
                         "since lambda is not transferable across measures")
    ap.add_argument("--vectors", type=Path, default=None,
                    help="node2vec prefix, for --assoc node2vec")
    ap.add_argument("--out", type=Path, default=config.WORK_DIR / "expb_report.json")
    args = ap.parse_args(argv)

    sel = json.loads(args.sweep.read_text())["best"]
    lam = sel["lambda"]
    graph = CoGraph.load(args.graph)
    ppmi = association(args.assoc, graph, sel["alpha"], sel["min_count"],
                       args.vectors)
    bins = degree_bins(graph)
    task_path = args.task or config.WORK_DIR / f"expb_task_{args.split}.jsonl"

    def iter_locations():
        """One location at a time: the full list is ~3.1 GB of id strings."""
        with open(task_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if args.limit and i >= args.limit:
                    return
                yield json.loads(line)

    if args.limit:
        n_locs = args.limit
    else:
        with open(task_path, encoding="utf-8") as f:
            n_locs = sum(1 for _ in f)
    # A directory of shards is loaded in RAM; an mmap prefix is not. The
    # 1.48M-candidate build only fits the second way on this machine.
    scorer = (MmapScorer(args.embeddings)
              if Path(f"{args.embeddings}_index.json").exists()
              else DenseScorer(args.embeddings))
    print(f"{n_locs} locations | lambda={lam} | seeds={args.seeds} | "
          f"base standardize={args.standardize}", flush=True)

    reveal = args.mode == "revealed"
    per_arm = {a: {k: [] for k in KS} for a in ARMS}
    per_rem = {a: {k: [] for k in KS} for a in ARMS}
    # Corpus-wide the gold is not guaranteed to be retrieved, so part of
    # G(l) - {seed} is unreachable by EVERY arm. Reported rather than
    # corrected for: it is a ceiling both arms share, and dividing by it
    # would leave the ratio between them unchanged while making the absolute
    # numbers look like something the reranker achieved.
    reach: list[float] = []
    sizes: list[int] = []
    groups: list[str] = []
    # Keyed by the anchor state the arm held ENTERING a step, so the curve
    # answers "given t anchors of which c were right, how often is the next
    # pick right" -- not the tautology "correct picks follow correct picks".
    curve: dict[tuple[int, int], list[int]] = {}
    seed_quality: dict[bool, list[float]] = {True: [], False: []}
    for seed in range(args.seeds):
        for i, loc in enumerate(iter_locations()):
            shared = base_scores(scorer, loc)
            shared_prep = ppmi.prepare(loc["candidates"])
            for arm in ARMS:
                trace = [] if arm == "hybrid_retrieved" else None
                r = run_location(loc, scorer, ppmi, graph, bins, lam, seed, arm,
                                 base=shared, reveal_seed=reveal, trace=trace,
                                 standardize=args.standardize,
                                 prep=shared_prep)
                for k in KS:
                    per_arm[arm][k].append(r[k])
                    if r.remaining is not None:
                        per_rem[arm][k].append(r.remaining[k])
                if trace:
                    for t in trace:
                        curve.setdefault(
                            (t["n_anchors"], t["n_correct"]), []
                        ).append(int(t["correct"]))
                    # In selfseed mode the seed IS the first pick, so its
                    # correctness is the seed quality. In revealed mode the
                    # seed is true by construction and the stratum is empty.
                    if not reveal:
                        seed_quality[trace[0]["correct"]].append(r[max(KS)])
            if reveal and len(loc["gold"]) > 1:
                cands = set(loc["candidates"])
                revealed = revelation_order(
                    loc["gold"], seed, f"{loc['citing_id']}:{loc['location']}"
                )[0]
                reach.append(
                    sum(1 for g in loc["gold"] if g != revealed and g in cands)
                    / (len(loc["gold"]) - 1)
                )
            groups.append(loc["citing_id"])
            sizes.append(len(loc["gold"]))
            if i and i % 500 == 0:
                print(f"  seed {seed}: {i}/{n_locs}", flush=True)

    report = {"split": args.split, "mode": args.mode,
              "task": str(task_path), "locations": n_locs,
              "seeds": args.seeds, "lambda": lam,
              "standardize": args.standardize, "n_boot": N_BOOT,
              "arms": {}, "deltas": {}}
    print(f"\n{'arm':<18} " + "  ".join(f"R@{k}".rjust(8) for k in KS))
    for arm in ARMS:
        vals = {k: paper_mean(per_arm[arm][k], groups) for k in KS}
        report["arms"][arm] = vals
        print(f"{arm:<18} " + "  ".join(f"{vals[k]:8.5f}" for k in KS))

    if per_rem["semantic"][KS[0]]:
        # Same run, scored over G(l) minus the revealed seed. See `Recalls`
        # for why the seed-inclusive table above cannot answer this.
        report["remaining_gold_reachability"] = sum(reach) / len(reach)
        report["remaining"] = {}
        base = {k: paper_mean(per_rem["semantic"][k], groups) for k in KS}
        print("\nremaining members only (revealed seed excluded); "
              f"reachable share {report['remaining_gold_reachability']:.5f}")
        print(f"{'arm':<18} " + "  ".join(f"R@{k}".rjust(8) for k in KS))
        for arm in ARMS:
            vals = {k: paper_mean(per_rem[arm][k], groups) for k in KS}
            report["remaining"][arm] = vals
            print(f"{arm:<18} " + "  ".join(f"{vals[k]:8.5f}" for k in KS))
            if arm != "semantic":
                print(f"{'  x semantic':<18} "
                      + "  ".join(f"{vals[k] / base[k]:7.3f}x" for k in KS))

        kmax = max(KS)
        report["remaining_by_set_size"] = {}
        print(f"\nremaining R@{kmax} by citations at the location")
        print(f"{'citations':>9} {'locations':>10} {'semantic':>10} "
              f"{'hybrid':>10} {'ratio':>8}")
        strata = {arm: by_set_size(per_rem[arm][kmax], groups, sizes)
                  for arm in ("semantic", "hybrid_retrieved")}
        for size, cell in strata["semantic"].items():
            hyb = strata["hybrid_retrieved"][size]["mean"]
            label = f"{size}" if size < 5 else ">=5"
            report["remaining_by_set_size"][label] = {
                "locations": cell["rows"] // args.seeds,
                "semantic": cell["mean"], "hybrid_retrieved": hyb,
            }
            print(f"{label:>9} {cell['rows'] // args.seeds:>10} "
                  f"{cell['mean']:>10.5f} {hyb:>10.5f} "
                  f"{hyb / cell['mean'] if cell['mean'] else float('nan'):>7.3f}x")

        report["remaining_deltas"] = {}
        print(f"\n{'remaining, paired vs semantic':<30} {'K':>3} "
              f"{'delta':>10}  95% CI")
        for arm in ARMS[1:]:
            report["remaining_deltas"][arm] = {}
            for k in KS:
                d = [x - y for x, y in zip(per_rem[arm][k],
                                           per_rem["semantic"][k], strict=True)]
                lo, hi = cluster_ci(d, groups, n_boot=N_BOOT, seed=config.SEED)
                m = paper_mean(d, groups)
                report["remaining_deltas"][arm][k] = {"delta": m, "ci": [lo, hi]}
                if k in (1, 5, 20):
                    print(f"{arm:<30} {k:>3} {m:>+10.5f}  [{lo:+.5f}, {hi:+.5f}]")

    print(f"\n{'paired vs semantic':<20} {'K':>3} {'delta':>10}  95% CI")
    for arm in ARMS[1:]:
        report["deltas"][arm] = {}
        for k in KS:
            d = [x - y for x, y in zip(per_arm[arm][k], per_arm["semantic"][k],
                                       strict=True)]
            lo, hi = cluster_ci(d, groups, n_boot=N_BOOT, seed=config.SEED)
            m = paper_mean(d, groups)
            report["deltas"][arm][k] = {"delta": m, "ci": [lo, hi]}
            if k in (1, 5, 20):
                print(f"{arm:<20} {k:>3} {m:>+10.5f}  [{lo:+.5f}, {hi:+.5f}]")

    # Completion accuracy as a function of the information state (the
    # seed-quality / information-state curve). Strata with too few
    # observations are dropped rather than reported at high variance.
    report["information_state"] = [
        {"n_anchors": t, "n_correct": c, "steps": len(v),
         "p_next_correct": sum(v) / len(v)}
        for (t, c), v in sorted(curve.items()) if len(v) >= 200
    ]
    print(f"\n{'anchors':>8} {'correct':>8} {'steps':>9} {'P(next ok)':>11}")
    for row in report["information_state"][:16]:
        print(f"{row['n_anchors']:>8} {row['n_correct']:>8} "
              f"{row['steps']:>9} {row['p_next_correct']:>11.5f}")

    if not reveal:
        report["seed_quality"] = {
            str(ok): {"locations": len(v),
                      "r_at_max_k": sum(v) / len(v) if v else None}
            for ok, v in seed_quality.items()
        }
        for ok, v in seed_quality.items():
            if v:
                print(f"seed {'correct' if ok else 'wrong  '}: "
                      f"{len(v):>7} locations, R@{max(KS)} {sum(v)/len(v):.5f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
