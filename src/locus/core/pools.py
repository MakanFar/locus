"""Frozen LOCUS-Rank pools and LOCUS-Swap pairs.

Every scorer sees identical pools, so score differences are attributable to
the scorer. Pool size is fixed rather than "all available distractors" so the
chance floor is constant across papers -- a paper with 12 references and one
with 300 must be comparable.
"""
import collections
import itertools
import random
from dataclasses import dataclass

from locus import config
from locus.core.types import ContextRec, gold_sets


@dataclass(frozen=True)
class RankItem:
    context_id: str
    citing_id: str
    gold: str
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class SwapPair:
    citing_id: str
    ctx_a: str
    ctx_b: str
    gold_a: str
    gold_b: str


def build_rank_items(
    index: dict[str, list[ContextRec]],
    pool_size: int = config.POOL_SIZE,
    seed: int = config.SEED,
) -> tuple[list[RankItem], collections.Counter]:
    out: list[RankItem] = []
    drops: collections.Counter = collections.Counter()
    for citing_id in sorted(index):
        recs = index[citing_id]
        golds = gold_sets(recs)
        all_refs = set().union(*golds.values()) if golds else set()
        for rec in recs:
            if not rec.alignment_ok:
                drops["alignment_failed"] += 1
                continue
            available = sorted(all_refs - golds[rec.location])
            if len(golds) < 2:
                drops["single_location"] += 1
                continue
            if len(available) < pool_size - 1:
                drops["too_few_distractors"] += 1
                continue
            rng = random.Random(f"{seed}:{rec.context_id}")
            distractors = rng.sample(available, pool_size - 1)
            cands = [*distractors, rec.refid]
            rng.shuffle(cands)
            drops["kept"] += 1
            out.append(
                RankItem(
                    context_id=rec.context_id,
                    citing_id=citing_id,
                    gold=rec.refid,
                    candidates=tuple(cands),
                )
            )
    return out, drops


def build_swap_pairs(
    index: dict[str, list[ContextRec]],
    k: int = config.SWAP_K,
    seed: int = config.SEED,
) -> list[SwapPair]:
    """Freeze up to k swap pairs per citing paper.

    Assumes each paper's records (`index[citing_id]`) are already sorted by
    `context_id`: `first.setdefault(rec.location, rec)` below picks, for each
    location, whichever record it meets first in `recs` to represent that
    location, so the result depends on that ordering. `corpus.build_index`
    sorts each paper's rows by `context_id` before constructing `ContextRec`s,
    which is what makes that choice deterministic.
    """
    out: list[SwapPair] = []
    for citing_id in sorted(index):
        recs = index[citing_id]
        golds = gold_sets(recs)
        first: dict[int, ContextRec] = {}
        for rec in recs:
            first.setdefault(rec.location, rec)
        eligible = [
            (a, b)
            for a, b in itertools.combinations(sorted(first), 2)
            if not (golds[a] & golds[b])
        ]
        if not eligible:
            continue
        rng = random.Random(f"{seed}:{citing_id}")
        chosen = eligible if len(eligible) <= k else rng.sample(eligible, k)
        for a, b in sorted(chosen):
            ra, rb = first[a], first[b]
            out.append(
                SwapPair(
                    citing_id=citing_id,
                    ctx_a=ra.context_id,
                    ctx_b=rb.context_id,
                    gold_a=ra.refid,
                    gold_b=rb.refid,
                )
            )
    return out
