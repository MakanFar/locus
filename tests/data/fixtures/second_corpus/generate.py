"""Generate a second, independently-built corpus for the JSONL contract.

Nothing here is derived from Gu et al.'s data: every id, filler word, and
citation pattern is synthesized from scratch by this script. It exists to
retire the risk the spec names in its own words -- "a contract validated
only against the corpus it was extracted from is a guess" -- by giving the
custom-corpus path (`locus.data.ingest`) a corpus that was never anywhere
near the code that defines the contract.

Shape. Each citing paper has `LOCATIONS_PER_PAPER` distinct citation sites
(e.g. "as shown in [1] and [2]"); each site groups `REFS_PER_LOCATION`
references cited together. One context row is emitted per reference at a
site, with that reference as TARGETCIT and its sitemates as OTHERCIT. That
is what gives every site multiple contexts to cluster (the clustering
evidence route in `locus.core.locations.cluster`) *and* multiple markers to
align (the marker evidence route in `locus.core.alignment`) -- neither route
is trivial the way a one-context-per-location corpus would make it.

Every location gets its own private block of filler vocabulary (see `_word`
below), so two contexts built from the same location share byte-identical
`raw` text (Jaccard 1.0 -- they cluster together) while two different
locations share zero words (Jaccard 0.0 -- they never accidentally merge),
regardless of `THETA`. Each context is checked against
`markers_with_numbers` as it is built, not assumed to align.

**Reference popularity is deliberately uneven.** A fraction of every paper's
reference slots is drawn from `N_SHARED` globally shared references instead of
its own private ones, so target frequencies across a split are not all 1. The
first version of this corpus gave every reference exactly one citation, which
made the popularity scorer a constant scorer -- and `export --verify`'s
`rank popularity` control, whose entire job is to show that the rank floor is
the tie rule rather than a stuck value, could not fire on it. A corpus where
that control cannot fire cannot demonstrate the export is sufficient, which is
the thing this fixture exists to demonstrate. Sharing is cross-paper only:
within a paper the references stay distinct, so `all_refs` per paper is
unchanged at `LOCATIONS_PER_PAPER * REFS_PER_LOCATION` and the maximum
supported pool size is still 7.

Deterministic given SEED: rerunning this script reproduces the committed
output byte for byte, which is what lets the generator be committed beside
its own output instead of the output standing alone.

Usage:  .venv/bin/python -m tests.data.fixtures.second_corpus.generate
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from locus.core.alignment import markers_with_numbers

OUT = Path(__file__).resolve().parent

SEED = 20260828

N_PAPERS = 40
LOCATIONS_PER_PAPER = 4
REFS_PER_LOCATION = 2
WORDS_PER_FILLER = 3  # words in each filler segment between/around markers

N_SHARED = 6      # globally shared references, so target counts are not all 1
SHARED_RATE = 0.25  # probability a reference slot draws a shared reference

TRAIN_PAPERS = 24
VAL_PAPERS = 8
# the remaining N_PAPERS - TRAIN_PAPERS - VAL_PAPERS papers go to test


def _word(n: int) -> str:
    """The n-th distinct lowercase-alphabetic token: a, b, ..., z, aa, ab, ...

    Pure letters only (no digits), because `locus.core.locations.shingles`
    tokenises on `[a-z]+` after lowercasing -- a digit-bearing token like
    "loc12" would split into two single-letter fragments and stop being a
    reliable, exclusive vocabulary block for one location.
    """
    n += 1
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(97 + r) + s
    return s


def _location_text(
    word_base: int, numbers_and_targets: list[tuple[int, bool]]
) -> tuple[str, str]:
    """Build (raw, masked_text) for one context at one location.

    `numbers_and_targets` is this location's full marker list -- (this
    paper's bibliography number, is_target) for every reference cited at
    this site, in a fixed order shared by every context built from this
    location. Only `is_target` varies between the contexts drawn from the
    same location, so they render byte-identical filler text and differ
    only in which marker is TARGETCIT.
    """
    n_fillers = len(numbers_and_targets) + 1
    fillers = [
        " ".join(
            _word(word_base + f * WORDS_PER_FILLER + w)
            for w in range(WORDS_PER_FILLER)
        )
        for f in range(n_fillers)
    ]
    raw_parts = [fillers[0]]
    masked_parts = [fillers[0]]
    for i, (num, is_target) in enumerate(numbers_and_targets):
        raw_parts.append(f"[{num}]")
        masked_parts.append("TARGETCIT" if is_target else "OTHERCIT")
        raw_parts.append(fillers[i + 1])
        masked_parts.append(fillers[i + 1])
    return " ".join(raw_parts), " ".join(masked_parts)


def build() -> dict:
    rng = random.Random(SEED)
    paper_order = list(range(N_PAPERS))
    rng.shuffle(paper_order)
    split_of: dict[int, str] = {}
    for rank, idx in enumerate(paper_order):
        if rank < TRAIN_PAPERS:
            split_of[idx] = "train"
        elif rank < TRAIN_PAPERS + VAL_PAPERS:
            split_of[idx] = "val"
        else:
            split_of[idx] = "test"

    contexts: list[dict] = []
    papers: list[dict] = [
        {
            "id": f"ref_shared_{i}",
            "title": f"Synthetic shared reference {i}",
            "abstract": "Synthetic abstract; not derived from any real corpus.",
        }
        for i in range(N_SHARED)
    ]
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    word_base = 0
    words_per_location = (REFS_PER_LOCATION + 1) * WORDS_PER_FILLER

    for p in range(N_PAPERS):
        citing_id = f"citing_{p:03d}"
        papers.append(
            {
                "id": citing_id,
                "title": f"Synthetic citing paper {p}",
                "abstract": "Synthetic abstract; not derived from any real corpus.",
            }
        )
        refs_per_paper = LOCATIONS_PER_PAPER * REFS_PER_LOCATION
        # Private references, with some slots replaced by shared ones. The
        # replacement is rejected if it would duplicate a reference already in
        # this paper: a paper citing the same work twice would collapse two
        # gold sets and shrink the distractor pool below what POOL_SIZE=7
        # needs.
        ref_ids = [f"ref_{p:03d}_{j}" for j in range(refs_per_paper)]
        for j in range(refs_per_paper):
            if rng.random() >= SHARED_RATE:
                continue
            candidate = f"ref_shared_{rng.randrange(N_SHARED)}"
            if candidate not in ref_ids:
                ref_ids[j] = candidate
        for j, refid in enumerate(ref_ids):
            if refid.startswith("ref_shared_"):
                continue  # emitted once, after the paper loop
            papers.append(
                {
                    "id": refid,
                    "title": f"Synthetic reference {p}-{j}",
                    "abstract": "Synthetic abstract; not derived from any real corpus.",
                }
            )

        # One bibliography number per reference, unique across the whole
        # paper (not just its location) -- exactly how a real paper's
        # reference list works, and what keeps build_bibliography from ever
        # seeing two different refids claim the same number.
        numbers = list(range(1, refs_per_paper + 1))
        rng.shuffle(numbers)

        for loc in range(LOCATIONS_PER_PAPER):
            site_refs = ref_ids[loc * REFS_PER_LOCATION : (loc + 1) * REFS_PER_LOCATION]
            site_numbers = numbers[loc * REFS_PER_LOCATION : (loc + 1) * REFS_PER_LOCATION]
            for t, target_ref in enumerate(site_refs):
                nts = [(n, k == t) for k, n in enumerate(site_numbers)]
                raw, masked = _location_text(word_base, nts)

                check = markers_with_numbers(raw, masked)
                if check is None:
                    raise SystemExit(
                        f"generated row for {citing_id}/{target_ref} failed to "
                        "align -- generator bug, not a corpus defect"
                    )
                targets = [num for typ, num in check if typ == "TARGETCIT"]
                if targets != [str(site_numbers[t])]:
                    raise SystemExit(
                        f"generated row for {citing_id}/{target_ref} resolved "
                        f"to target numbers {targets!r}, expected "
                        f"[{site_numbers[t]!r}]"
                    )

                context_id = f"ctx_{p:03d}_{loc}_{t}"
                contexts.append(
                    {
                        "context_id": context_id,
                        "citing_id": citing_id,
                        "refid": target_ref,
                        "raw": raw,
                        "masked_text": masked,
                    }
                )
                splits[split_of[p]].append(context_id)
            word_base += words_per_location

    return {"contexts": contexts, "papers": papers, "splits": splits}


def write(data: dict) -> None:
    with open(OUT / "contexts.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in data["contexts"])
    with open(OUT / "papers.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in data["papers"])
    (OUT / "splits").mkdir(exist_ok=True)
    for name, ids in data["splits"].items():
        (OUT / "splits" / f"{name}.json").write_text(json.dumps(ids))
    print(
        f"papers={N_PAPERS} contexts={len(data['contexts'])} "
        f"train={len(data['splits']['train'])} val={len(data['splits']['val'])} "
        f"test={len(data['splits']['test'])}"
    )


def main() -> None:
    write(build())


if __name__ == "__main__":
    main()
