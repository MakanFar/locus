"""Extract the texts SPECTER2 must embed to score a frozen LOCUS split.

The pools are frozen, so the scoring workload is not the corpus -- it is the
union of the candidates that appear in some pool and the contexts that pose
some query. On test that is 37,409 papers and 76,293 contexts, so the 7.4 GB
corpus never has to leave this machine: this writes one parquet of ~114k rows,
which the GPU job embeds in minutes.

Two sides, two recipes:

  kind="paper"    f"{title} [SEP] {abstract}"   the pinned production recipe
                  (allenai/specter2_base @ 3447645e, CLS pooling, 512 tokens),
                  reproduced here by locus.scoring.embed and verified against
                  the production vectors it was copied from
  kind="context"  the masked window, alone

The context side is the masked window and nothing else. The plan lists
siblings-as-text (q = l + titles(A)) as a *control*, which only
means something if the base query is the location's text by itself.

**No abstract-length gate.** The production recipe skips abstracts of 50
characters or fewer; here that would be a correctness bug rather than a
saving. A candidate with no vector cannot be scored, so its pool would silently
shrink from 10 to 9 -- and the chance floor is H_k/k, which *moves* with k
(0.29290 at 10, 0.31433 at 9). The frozen floor the build gate pins would no
longer be the floor of the pools actually scored. Abstract-less candidates are
therefore embedded from their title alone, and counted in the manifest.

Usage:
    .venv/bin/python -m locus.build.embed_inputs --split test
    .venv/bin/python -m locus.build.embed_inputs --corpus DIR --split test

`--corpus DIR` takes the paper titles and abstracts from a JSONL corpus in
the contract `locus.data.ingest` documents rather than from the upstream
`papers.json`. It has to exist for the same reason `build.pipeline --corpus`
does: this module calls `load_papers`, so without it a custom corpus could be
built and pooled and then had no way to reach embeddings, and therefore no way
to reach `eval` or `verify` either. The masked windows come from
`index_{split}.pkl`, which already carries them whichever corpus built it.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys

import polars as pl

from locus import config
from locus.core.types import anchors_union, gold_sets
from locus.data.corpus import load_index, load_papers

PAPER_RECIPE = "title [SEP] abstract; max_length=512; CLS pooling"
CONTEXT_RECIPE = "masked window; max_length=512; CLS pooling"
SIBLINGS_RECIPE = (
    "masked window + titles of A(l), anchor order sorted by id; "
    "max_length=512; CLS pooling"
)


def compose_paper_text(title: str, abstract: str) -> str:
    """The production recipe, literal '[SEP]', empty title or abstract allowed."""
    return f"{title} [SEP] {abstract}"


def collect_keys(rank_items, swap_pairs) -> tuple[set[str], set[str]]:
    """Every paper that needs a vector, and every context that poses a query."""
    papers: set[str] = set()
    contexts: set[str] = set()
    for item in rank_items:
        papers.update(item.candidates)
        contexts.add(item.context_id)
    for pair in swap_pairs:
        papers.add(pair.gold_a)
        papers.add(pair.gold_b)
        contexts.add(pair.ctx_a)
        contexts.add(pair.ctx_b)
    return papers, contexts


def build_frame(
    split: str, variant: str = "plain", *, corpus: str | None = None
) -> tuple[pl.DataFrame, dict]:
    paper_source = None
    if corpus is not None:
        from locus.data.ingest import jsonl_papers

        paper_source = jsonl_papers(corpus)
    work = config.WORK_DIR
    with open(work / f"rank_items_{split}.pkl", "rb") as f:
        rank_items = pickle.load(f)
    with open(work / f"swap_pairs_{split}.pkl", "rb") as f:
        swap_pairs = pickle.load(f)
    index = load_index(work / f"index_{split}.pkl")

    paper_ids, context_ids = collect_keys(rank_items, swap_pairs)
    papers = load_papers(paper_ids, source=paper_source)

    by_ctx = {r.context_id: r for recs in index.values() for r in recs}
    missing = sorted(context_ids - set(by_ctx))
    if missing:
        raise SystemExit(
            f"{len(missing)} pooled contexts are absent from index_{split}.pkl, "
            f"e.g. {missing[:5]} — the pools and the index disagree, so they "
            "were not built from the same run"
        )

    # The siblings-as-text control appends the anchors' TITLES to
    # the query and changes nothing else, so it isolates one question: can the
    # rescorer's PPMI term be replaced by simply telling the encoder what else
    # is cited here? Anchors are emitted in sorted order -- an anchor set is a
    # frozenset, and iterating one directly would make the query text depend on
    # PYTHONHASHSEED.
    anchor_of: dict[str, frozenset[str]] = {}
    if variant == "siblings":
        for recs in index.values():
            golds = gold_sets(recs)
            for rec in recs:
                if rec.context_id in context_ids:
                    anchor_of[rec.context_id] = frozenset(anchors_union(rec, golds))
        anchor_titles = load_papers(
            {a for s_ in anchor_of.values() for a in s_} - set(papers),
            source=paper_source,
        )
        anchor_titles.update(papers)

    def context_text(cid: str) -> str:
        base = by_ctx[cid].masked
        if variant != "siblings":
            return base
        titles = [anchor_titles[a]["title"] for a in sorted(anchor_of.get(cid, ()))
                  if anchor_titles.get(a, {}).get("title")]
        return base + (" " + " ".join(titles) if titles else "")

    rows = [
        {"key": pid, "kind": "paper",
         "text": compose_paper_text(papers[pid]["title"], papers[pid]["abstract"])}
        for pid in sorted(paper_ids)
    ]
    rows += [
        {"key": cid, "kind": "context", "text": context_text(cid)}
        for cid in sorted(context_ids)
    ]
    frame = pl.DataFrame(rows, schema={"key": pl.Utf8, "kind": pl.Utf8, "text": pl.Utf8})

    no_abstract = sum(1 for pid in paper_ids if not papers[pid]["abstract"])
    no_title = sum(1 for pid in paper_ids if not papers[pid]["title"])
    empty_context = sum(1 for cid in context_ids if not by_ctx[cid].masked.strip())
    if empty_context:
        # An empty query embeds to the CLS vector of a bare [CLS][SEP] pair --
        # the same vector for every such context, which would score its whole
        # pool identically and quietly park those items on the tie floor.
        raise SystemExit(
            f"{empty_context} of {len(context_ids)} contexts have an empty "
            "masked window; they would all embed to one degenerate vector"
        )

    with_siblings = sum(1 for c in context_ids if anchor_of.get(c)) if variant == "siblings" else 0
    manifest = {
        "split": split,
        "variant": variant,
        "corpus": corpus,
        "contexts_with_sibling_titles": with_siblings,
        "papers": len(paper_ids),
        "contexts": len(context_ids),
        "rows": frame.height,
        "papers_without_abstract": no_abstract,
        "papers_without_title": no_title,
        "paper_text_recipe": PAPER_RECIPE,
        "context_text_recipe": SIBLINGS_RECIPE if variant == "siblings" else CONTEXT_RECIPE,
        "rank_items": len(rank_items),
        "swap_pairs": len(swap_pairs),
    }
    return frame, manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("test", "val"), default="test")
    ap.add_argument("--variant", choices=("plain", "siblings"), default="plain")
    ap.add_argument(
        "--corpus", default=None,
        help="a JSONL corpus directory (locus.data.ingest); without it the "
             "upstream papers.json at $LOCUS_DATA_DIR is used",
    )
    args = ap.parse_args(argv)

    frame, manifest = build_frame(args.split, args.variant, corpus=args.corpus)
    suffix = "" if args.variant == "plain" else f"_{args.variant}"
    out = config.WORK_DIR / f"embed_inputs{suffix}_{args.split}.parquet"
    frame.write_parquet(out)
    man_path = config.WORK_DIR / f"embed_inputs{suffix}_{args.split}.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {man_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
