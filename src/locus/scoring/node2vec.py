"""node2vec over the train-only co-citation graph, and the two things it is for.

The reranker's structural term reads the co-citation graph through a
hand-specified statistic, PPMI. The obvious question is whether a learned
graph representation reads it better. node2vec is the standard answer, and it
lands in two quite different places here.

**As a base retriever it is a null control, not a baseline.** The base scorer
must produce S(c | l): a score for a candidate given a citation *context*.
node2vec embeds nodes of a graph, and a citation context is not a node --- the
nearest available query is the citing paper, which is identical for every
location in it. Lemma 1 then pins any such scorer to exactly H_k/k on
LOCUS-Rank and exactly 0.5 on LOCUS-Swap. That is worth running precisely
because node2vec is strong: it makes the floor a statement about what the
benchmark measures rather than about how weak the null scorer was.

**As an association measure it is a real comparison**, replacing PPMI(c,a)
with the cosine between the two papers' node vectors, with lambda re-swept on
validation because the two live on different scales.

Two construction choices that are not cosmetic:

  *Positive cosine.* Similarities are clipped at zero, matching PPMI's own
  max(0, .). Without the clip a pair absent from the graph would score 0.0 and
  so rank *above* a pair the embedding judged actively dissimilar, which
  inverts the meaning of "no evidence".

  *Same coverage rules as PPMI.* 20.4% of the graph's 496,008 nodes are
  isolated and generate no walks, and 18.84% of test rank candidates are not
  in the train vocabulary at all. Both score 0.0, exactly as PPMI scores an
  unseen pair, so the arms differ in how the graph is read and not in which
  pairs they can see.

Usage:
    .venv/bin/python -m locus.scoring.node2vec --walks 10 --length 40 \\
        --p 1.0 --q 1.0 --dim 128
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from locus import config
from locus.core.protocols import Association
from locus.scoring.cocitation import CoGraph


def edge_keys(counts) -> np.ndarray:
    """Sorted int64 keys `row * N + col`, for vectorised edge lookup.

    node2vec's second-order bias needs "is x a neighbour of the previous node
    t", once per walker per step. A per-walker binary search into a ragged CSR
    row cannot be vectorised; a single sorted array of global edge keys can, so
    the whole walk advances in lockstep instead of in a Python loop over
    millions of walkers.
    """
    coo = counts.tocoo()
    keys = coo.row.astype(np.int64) * counts.shape[0] + coo.col.astype(np.int64)
    keys.sort()
    return keys


def _is_edge(keys: np.ndarray, src: np.ndarray, dst: np.ndarray,
             n: int) -> np.ndarray:
    probe = src.astype(np.int64) * n + dst.astype(np.int64)
    pos = np.searchsorted(keys, probe)
    pos = np.minimum(pos, len(keys) - 1)
    return keys[pos] == probe


def walks(counts, p: float, q: float, num_walks: int, length: int,
          seed: int, progress=None) -> np.ndarray:
    """`num_walks` biased walks of `length` from every non-isolated node.

    Rejection sampling, as in the reference implementation: propose a uniform
    neighbour and accept with probability w/wmax, where w is 1/p for a return
    to the previous node, 1 for a neighbour of it, and 1/q otherwise. The
    alternative -- per-edge alias tables -- costs sum-of-squared-degrees
    memory, which this graph's degree tail (max 1,890) makes unattractive.
    """
    n = counts.shape[0]
    indptr, indices = counts.indptr, counts.indices
    deg = np.diff(indptr)
    start = np.flatnonzero(deg > 0).astype(np.int32)
    keys = edge_keys(counts)
    wmax = max(1.0, 1.0 / p, 1.0 / q)
    rng = np.random.default_rng(seed)

    out = np.full((num_walks * len(start), length), -1, dtype=np.int32)
    for w in range(num_walks):
        cur = start.copy()
        row = out[w * len(start):(w + 1) * len(start)]
        row[:, 0] = cur
        prev = None
        for step in range(1, length):
            d = deg[cur]
            if prev is None:
                nxt = indices[indptr[cur] + rng.integers(0, d)]
            else:
                nxt = np.empty(len(cur), dtype=indices.dtype)
                todo = np.arange(len(cur))
                # Bounded retries: a walker that keeps being rejected takes the
                # last proposal. wmax is at most 2 for the grids we sweep, so
                # the acceptance rate is >= 0.5 and 12 rounds leave a rejection
                # probability under 1e-3 -- and taking the proposal is a
                # uniform-neighbour step, not an invalid one.
                for _ in range(12):
                    if not len(todo):
                        break
                    c, t = cur[todo], prev[todo]
                    x = indices[indptr[c] + rng.integers(0, deg[c])]
                    weight = np.where(
                        x == t, 1.0 / p,
                        np.where(_is_edge(keys, t, x, n), 1.0, 1.0 / q))
                    keep = rng.random(len(todo)) < (weight / wmax)
                    nxt[todo] = x
                    todo = todo[~keep]
            prev, cur = cur, nxt.astype(np.int32)
            row[:, step] = cur
        if progress:
            progress(w + 1, num_walks)
    return out


def write_corpus(walk_array: np.ndarray, vocab: tuple[str, ...], path: Path,
                 chunk: int = 100_000) -> int:
    """One walk per line, paper ids as tokens, for gensim's corpus_file mode.

    Streaming through a file rather than holding the token lists in memory is
    the difference between a few hundred MB and several GB, and corpus_file is
    also gensim's only fully multi-threaded path.

    The id lookup is done a chunk at a time for the same reason. Indexing an
    object array by the whole walk matrix at once materialises 158M pointers
    -- about 1.3 GB before a single line is written -- to produce output that
    is consumed strictly in order.

    Written uncompressed on purpose: gensim's `corpus_file` path refuses a
    compressed file outright, and it is the only fully multi-threaded one. The
    file is large and disposable, so `main` deletes it once the vectors exist.
    """
    lookup = np.asarray(vocab, dtype=object)
    with open(path, "w", encoding="utf-8") as f:
        for lo in range(0, len(walk_array), chunk):
            block = lookup[walk_array[lo:lo + chunk]]
            f.write("\n".join(" ".join(line) for line in block))
            f.write("\n")
    return len(walk_array)


@dataclass(frozen=True)
class NodeVectors:
    """A PPMI-shaped view over node embeddings.

    Duck-types `cocitation.PPMI` -- `index`, `value`, `mean_over`, `aggregate`
    -- so every caller that reads an association measure works unchanged and
    the arms differ only in what the measure is.
    """

    vectors: np.ndarray          # L2-normalised, row i is vocab[i]
    index: dict[str, int]

    @classmethod
    def load(cls, prefix: Path) -> NodeVectors:
        vecs = np.load(f"{prefix}_vectors.npy")
        index = json.loads(Path(f"{prefix}_index.json").read_text())
        return cls(vectors=vecs, index=index)

    def value(self, c: str, a: str) -> float:
        i, j = self.index.get(c), self.index.get(a)
        if i is None or j is None:
            return 0.0
        return max(0.0, float(self.vectors[i] @ self.vectors[j]))

    def aggregate(self, c: str, anchors: frozenset[str],
                  how: str = "mean") -> float:
        if how not in ("mean", "masked_mean", "max"):
            raise ValueError(f"unknown aggregator {how!r}")
        if not anchors:
            return 0.0
        i = self.index.get(c)
        if i is None:
            return 0.0
        js = sorted(j for a in anchors if (j := self.index.get(a)) is not None)
        if not js:
            return 0.0
        sims = np.maximum(0.0, self.vectors[js] @ self.vectors[i])
        nz = sims[sims > 0.0]
        if how == "max":
            return float(sims.max())
        if how == "masked_mean":
            return float(nz.mean()) if len(nz) else 0.0
        # Divide by the full |A|, matching PPMI.aggregate: an anchor with no
        # node is evidence we do not have, not evidence that does not exist.
        return float(sims.sum() / len(anchors))

    def prepare(self, candidates: list[str]) -> np.ndarray:
        """Candidate vectors as a dense matrix, zero-filled where unknown.

        A zero row makes the dot product zero, which the positive clip in
        `add_to` then leaves at zero -- the same "no evidence" value PPMI
        gives a pair it has never seen, so the two measures differ in what
        they say about a pair and not in which pairs they can speak about.
        """
        rows = np.fromiter((self.index.get(c, -1) for c in candidates),
                           dtype=np.int64, count=len(candidates))
        out = np.zeros((len(candidates), self.vectors.shape[1]),
                       dtype=self.vectors.dtype)
        known = rows >= 0
        out[known] = self.vectors[rows[known]]
        return out

    def add_to(self, sums, anchor: str, prep) -> None:
        """Accumulate this anchor's similarity to every candidate at once.

        The sparse counterpart walks one CSR row; here every candidate has a
        vector, so the whole column of similarities is one matrix-vector
        product.
        """
        j = self.index.get(anchor)
        if j is None:
            return
        sums += np.maximum(0.0, prep @ self.vectors[j])

    def mean_over(self, c: str, anchors: frozenset[str]) -> float:
        return self.aggregate(c, anchors, "mean")


ASSOC_CHOICES = ("ppmi", "count", "node2vec")


def association(assoc: str, graph, alpha, min_count, prefix=None) -> Association:
    """Build the association view the reranker reads, by name.

    One factory rather than the same three-way branch in sweep, rank and swap:
    the arms must differ only in the measure, and three hand-copied branches
    is exactly how one of them ends up reading a different graph.

    `alpha` and `min_count` are PPMI's smoothing parameters and have no
    meaning for node2vec, whose measure is fixed once the vectors are trained.
    Callers sweeping a grid should collapse those axes for it rather than
    running the same embedding five times and reporting whichever tied first.
    """
    if assoc == "ppmi":
        return graph.ppmi(alpha=alpha, min_count=min_count)
    if assoc == "count":
        return graph.raw_counts(min_count=min_count)
    if assoc == "node2vec":
        return NodeVectors.load(
            prefix or (config.WORK_DIR / "node2vec_p1q1"))
    raise ValueError(f"unknown association measure {assoc!r}")


def null_control(prefix: Path, split: str, seed: int) -> dict:
    """Run the embedding as a base scorer against the frozen pools and pairs.

    Reported as an assertion on Swap (exactly 0.5, zero-width interval) and as
    a measurement on Rank, for the reason in
    `baselines.graph_embedding_scorer`.
    """
    import pickle

    from locus.core.bootstrap import cluster_ci
    from locus.scoring.baselines import (
        graph_embedding_scorer,
        rank_mrr,
        swap_accuracy,
    )

    W = config.WORK_DIR
    with open(W / f"rank_items_{split}.pkl", "rb") as f:
        items = pickle.load(f)
    with open(W / f"swap_pairs_{split}.pkl", "rb") as f:
        pairs = pickle.load(f)
    nv = NodeVectors.load(prefix)
    citing_of = {it.context_id: it.citing_id for it in items}
    for p in pairs:
        citing_of.setdefault(p.ctx_a, p.citing_id)
        citing_of.setdefault(p.ctx_b, p.citing_id)

    scorer = graph_embedding_scorer(nv, citing_of)
    floor = sum(1.0 / i for i in range(1, config.POOL_SIZE + 1)) / config.POOL_SIZE
    mrr, _values, _groups = rank_mrr(items, scorer)
    acc, outcomes, sgroups = swap_accuracy(pairs, scorer)
    lo, hi = cluster_ci(outcomes, sgroups, n_boot=1000, seed=seed)
    return {
        "prefix": str(prefix), "split": split,
        "rank": {"mrr": mrr, "floor": floor, "moves_off_floor": mrr != floor,
                 "n": len(items)},
        "swap": {"accuracy": acc, "ci": [lo, hi],
                 "pinned_exactly": acc == 0.5 and set(outcomes) == {0.5}
                 and lo == 0.5 and hi == 0.5, "n": len(pairs)},
    }


def main(argv: list[str] | None = None) -> int:
    from gensim.models import Word2Vec

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", type=Path,
                    default=config.WORK_DIR / "cocitation_train.npz")
    ap.add_argument("--walks", type=int, default=10)
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--null-control", type=Path, default=None,
                    help="skip training; score this vector prefix as a base "
                         "scorer against the frozen pools and pairs")
    ap.add_argument("--split", default="test")
    ap.add_argument("--keep-corpus", action="store_true",
                    help="keep the (large, uncompressed) walk file")
    args = ap.parse_args(argv)

    if args.null_control is not None:
        r = null_control(args.null_control, args.split, args.seed)
        print(f"rank MRR {r['rank']['mrr']!r} against floor "
              f"{r['rank']['floor']!r} -> "
              f"{'moves off the floor' if r['rank']['moves_off_floor'] else 'ON THE FLOOR'}")
        print(f"swap acc {r['swap']['accuracy']!r} CI "
              f"[{r['swap']['ci'][0]!r}, {r['swap']['ci'][1]!r}] -> "
              f"{'pinned exactly at 0.5' if r['swap']['pinned_exactly'] else 'NOT PINNED'}")
        out = config.WORK_DIR / f"node2vec_null_control_{args.split}.json"
        out.write_text(json.dumps(r, indent=2))
        print(f"wrote {out}")
        return 0 if r["swap"]["pinned_exactly"] else 1

    tag = args.tag or f"p{args.p:g}q{args.q:g}"
    out = config.WORK_DIR / f"node2vec_{tag}"
    graph = CoGraph.load(args.graph)
    print(f"graph: {len(graph.vocab)} nodes, {graph.counts.nnz} non-zeros")

    t0 = time.time()
    W = walks(graph.counts, args.p, args.q, args.walks, args.length, args.seed,
              progress=lambda i, n: print(f"  walk {i}/{n} "
                                          f"({time.time()-t0:.0f}s)", flush=True))
    print(f"{W.shape[0]} walks x {W.shape[1]} in {time.time()-t0:.0f}s")

    corpus = config.WORK_DIR / f"node2vec_{tag}_walks.txt"
    write_corpus(W, graph.vocab, corpus)
    print(f"wrote {corpus}")

    t1 = time.time()
    model = Word2Vec(corpus_file=str(corpus), vector_size=args.dim, sg=1,
                     hs=0, negative=5, window=args.window, min_count=1,
                     workers=args.workers, epochs=args.epochs, seed=args.seed)
    print(f"trained {len(model.wv)} vectors in {time.time()-t1:.0f}s")

    keys = list(model.wv.index_to_key)
    V = np.asarray(model.wv[keys], dtype=np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True).clip(min=1e-12)
    np.save(f"{out}_vectors.npy", V)
    Path(f"{out}_index.json").write_text(
        json.dumps({k: i for i, k in enumerate(keys)}))
    meta = {"tag": tag, "p": args.p, "q": args.q, "dim": args.dim,
            "walks": args.walks, "length": args.length, "window": args.window,
            "epochs": args.epochs, "seed": args.seed,
            "graph_nodes": len(graph.vocab), "trained_vectors": len(keys),
            "isolated_nodes": int((np.diff(graph.counts.indptr) == 0).sum())}
    Path(f"{out}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}_vectors.npy ({V.shape}) and index/meta")
    if not args.keep_corpus:
        corpus.unlink()
        print(f"removed {corpus}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
