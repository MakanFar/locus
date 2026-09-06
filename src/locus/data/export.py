"""Portable, ID-only export of the frozen LOCUS artefacts.

The harness reads pickles: `rank_items_{split}.pkl`, `swap_pairs_{split}.pkl`,
`hard_ids_{split}.pkl`, `headline_ids_{split}.pkl`, and `index_{split}.pkl`.
That is right for us and wrong for anyone else, for three reasons:

  * A pickle is version-coupled to the dataclass that wrote it. Rename a field
    in `pools.RankItem` and every published copy stops loading.
  * Unpickling executes code, so "download our benchmark" is a request the
    recipient should refuse.
  * `A(l)` is not in the pool artefacts at all. It is recomputed on demand by
    `core.types.anchors_union` from `index_{split}.pkl`, which is 103 MB on test
    and carries the raw context windows -- so reproducing a rescoring result
    currently means shipping the upstream corpus text to get at a set of id
    strings.

This module writes the same information as JSONL and JSON, carrying **no text
of any kind** -- only paper ids, context ids, integers and booleans. That is
the property that makes it redistributable: the ids are ours to publish, the
abstracts and context windows are Gu et al.'s.

Four files per split:

  locus_rank_{split}.jsonl      one row per LOCUS-Rank pool
  locus_swap_{split}.jsonl      one row per LOCUS-Swap pair
  locus_anchors_{split}.jsonl   one row per context, the A(l) sidecar
  locus_manifest_{split}.json   counts, frozen config, floors, sha256s

**Why anchors are a separate file rather than a column on the rank rows.**
Duplicating them would make `locus_rank` self-sufficient, and it is tempting.
But the anchor sets are also needed for contexts that have *no* rank pool --
`build_rank_items` drops alignment failures and whole papers with too few
distractors -- and sequential completion keys on locations, not on rank items. One row
per context is the only shape that covers all three consumers, and the join is
a single dict lookup on `context_id`.

The anchors file also carries each context's own `gold`, which is not
redundant: the popularity null control counts targets over *every context in
the index*, not over rank-item golds (see `build_day1`, step 6 -- scoping it
to rank items collapses popularity into the constant scorer on a third of the
swap pairs). Without that column the exported benchmark could not reproduce
its own null controls, which is the one thing it most needs to be able to do.

Usage:
    .venv/bin/python -m locus.data.export --split test
    .venv/bin/python -m locus.data.export --split test --verify
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pickle
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

from locus import config
from locus.core.pools import RankItem, SwapPair
from locus.core.types import anchors, gold_sets
from locus.data.corpus import load_index

SCHEMA = 1


def rank_rows(items: Iterable[RankItem], hard: set[str],
              headline: set[str]) -> Iterator[dict]:
    """One row per rank pool, with the two frozen slices as flags.

    `candidates` keeps the frozen shuffled order. Sorting it would not change
    any metric -- scoring is a dict keyed by candidate -- but it would make the
    exported pool a different artefact from the one the paper's numbers were
    computed on, and there would be no way to tell from the file.
    """
    for it in items:
        yield {
            "context_id": it.context_id,
            "citing_id": it.citing_id,
            "gold": it.gold,
            "candidates": list(it.candidates),
            "hard": it.context_id in hard,
            "headline": it.context_id in headline,
        }


def swap_rows(pairs: Iterable[SwapPair]) -> Iterator[dict]:
    for p in pairs:
        yield {
            "citing_id": p.citing_id,
            "ctx_a": p.ctx_a,
            "ctx_b": p.ctx_b,
            "gold_a": p.gold_a,
            "gold_b": p.gold_b,
        }


def anchor_rows(index: dict) -> Iterator[dict]:
    """One row per context: A(l) by both routes, plus its own target.

    Both routes are reported alongside the union because neither subsumes the
    other, and which one dominates is an artefact of the extraction pipeline.
    A consumer that silently used only the clustering route would be running a
    different benchmark.
    """
    for citing_id in sorted(index):
        recs = sorted(index[citing_id], key=lambda r: r.context_id)
        golds = gold_sets(recs)
        for rec in recs:
            clustering = anchors(rec, golds)
            marker = set(rec.marker_anchors)
            yield {
                "context_id": rec.context_id,
                "citing_id": citing_id,
                "location": rec.location,
                "gold": rec.refid,
                "alignment_ok": rec.alignment_ok,
                "anchors": sorted(clustering | marker),
                "anchors_clustering": sorted(clustering),
                "anchors_marker": sorted(marker),
            }


def write_jsonl(path: Path, rows: Iterable[dict]) -> tuple[int, str]:
    """Write rows and return (count, sha256). Hashed while writing, not after.

    Re-reading the file to hash it would check that the disk agrees with the
    disk. Hashing the bytes on their way out checks that the file agrees with
    what this process meant to write.
    """
    h = hashlib.sha256()
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False) + "\n"
            f.write(line)
            h.update(line.encode("utf-8"))
            n += 1
    return n, h.hexdigest()


def load_rank(path: Path) -> list[RankItem]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append(RankItem(
                context_id=d["context_id"], citing_id=d["citing_id"],
                gold=d["gold"], candidates=tuple(d["candidates"]),
            ))
    return out


def load_swap(path: Path) -> list[SwapPair]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append(SwapPair(
                citing_id=d["citing_id"], ctx_a=d["ctx_a"], ctx_b=d["ctx_b"],
                gold_a=d["gold_a"], gold_b=d["gold_b"],
            ))
    return out


def load_anchors(path: Path) -> dict[str, frozenset[str]]:
    """context_id -> A(l), the union route. The drop-in for `anchors_union`."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["context_id"]] = frozenset(d["anchors"])
    return out


def load_target_counts(path: Path) -> collections.Counter:
    """Target frequencies over every context, for the popularity null."""
    counts: collections.Counter = collections.Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            counts[json.loads(line)["gold"]] += 1
    return counts


def rank_floor(pool_size: int) -> float:
    return sum(1.0 / i for i in range(1, pool_size + 1)) / pool_size


def pool_size_of(items: Iterable[RankItem]) -> int:
    """The single pool size shared by every item, or a hard error.

    The floor is H_k/k and it MOVES with k -- 0.29290 at 10, 0.31433 at 9 --
    so a set of pools with mixed sizes has no single floor to publish, and a
    manifest that named one would be wrong for some of its own rows. Taking k
    from `config.POOL_SIZE` instead would be worse still: the manifest would
    describe our configuration rather than the file the consumer holds, and
    the two can differ precisely when it matters.
    """
    sizes = {len(it.candidates) for it in items}
    if len(sizes) != 1:
        raise ValueError(
            f"rank pools have mixed sizes {sorted(sizes)}; the chance floor "
            "H_k/k is defined only for a single k, so this set cannot be "
            "exported with a floor"
        )
    return sizes.pop()


def _slices(split: str, work: Path) -> tuple[set[str], set[str]]:
    hard = set(pickle.loads((work / f"hard_ids_{split}.pkl").read_bytes()))
    headline = set(
        pickle.loads((work / f"headline_ids_{split}.pkl").read_bytes())
    )
    return hard, headline


def export(split: str, work: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    with open(work / f"rank_items_{split}.pkl", "rb") as f:
        items = pickle.load(f)
    with open(work / f"swap_pairs_{split}.pkl", "rb") as f:
        pairs = pickle.load(f)
    hard, headline = _slices(split, work)
    index = load_index(work / f"index_{split}.pkl")
    k = pool_size_of(items)

    files = {}
    n, sha = write_jsonl(out / f"locus_rank_{split}.jsonl",
                         rank_rows(items, hard, headline))
    files[f"locus_rank_{split}.jsonl"] = {"rows": n, "sha256": sha}
    n, sha = write_jsonl(out / f"locus_swap_{split}.jsonl", swap_rows(pairs))
    files[f"locus_swap_{split}.jsonl"] = {"rows": n, "sha256": sha}
    n, sha = write_jsonl(out / f"locus_anchors_{split}.jsonl",
                         anchor_rows(index))
    files[f"locus_anchors_{split}.jsonl"] = {"rows": n, "sha256": sha}

    manifest = {
        "schema": SCHEMA,
        "split": split,
        "source": {
            "dataset": "Local Citation Recommendation, arXiv split "
                       "(Gu et al., ECIR 2022)",
            "note": "This export carries paper and context IDENTIFIERS only. "
                    "Titles, abstracts and context windows are not included "
                    "and must be obtained from the upstream dataset.",
        },
        "config": {
            "pool_size": config.POOL_SIZE, "swap_k": config.SWAP_K,
            "shingle_n": config.SHINGLE_N, "theta": config.THETA,
            "hardness_tau": config.HARDNESS_TAU, "seed": config.SEED,
        },
        "floors": {
            # Exact, not rounded: these are assertions a consumer checks
            # against, and the rank floor moves with pool size (0.29290 at
            # k=10, 0.31433 at k=9), so a filtered pool is detectable.
            "pool_size": k,
            "rank_constant_mrr": repr(rank_floor(k)),
            "swap_context_independent_accuracy": "0.5",
        },
        "counts": {
            "rank_items": len(items), "swap_pairs": len(pairs),
            "contexts": sum(len(v) for v in index.values()),
            "citing_papers": len(index),
            "hard": len(hard), "headline": len(headline),
        },
        "files": files,
    }
    path = out / f"locus_manifest_{split}.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("test", "val"), default="test")
    ap.add_argument("--work", type=Path, default=config.WORK_DIR)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verify", action="store_true",
                    help="re-read the export and check it against the pickles "
                         "and its own null controls")
    args = ap.parse_args(argv)
    out = args.out or args.work / "export"

    if not args.verify:
        m = export(args.split, args.work, out)
        for name, meta in m["files"].items():
            print(f"  {name:34s} {meta['rows']:>7d} rows  {meta['sha256'][:12]}")
        print(f"  locus_manifest_{args.split}.json")
        print(f"\nwrote {out}")

    from locus.eval.verify import verify_export  # deferred: eval sits above data

    print("\nverifying")
    failures = verify_export(args.split, args.work, out)
    for f in failures:
        print(f"  FAIL {f}")
    if failures:
        return 1
    print("  round-trip OK, checksums OK, null controls exact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
