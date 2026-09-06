"""Okapi BM25 -- the lexical base scorer.

S_base(c | l) = BM25(masked window, candidate title+abstract), over exactly the
`(key, kind, text)` parquet `build/embed_inputs.py` already writes for the
dense encoders. Same query, same document text, same frozen pools: the only
thing that changes is how a (context, candidate) pair is scored, which is what
makes the comparison against `dense.DenseScorer` a fair one.

This is a base to be beaten, not a harness assertion -- which is why it does
not live in `baselines.py`, whose scorers are context-independent by design and
pinned to the floor by Lemma 1. BM25 reads the query, so nothing pins it.

Three choices that a reader has to be able to check, since BM25 is a family
rather than a formula:

  idf     log(1 + (N - n + 0.5) / (n + 0.5)), the Lucene form. The textbook
          Robertson/Sparck-Jones idf omits the +1 and goes NEGATIVE for a term
          in more than half the corpus, which would penalise a document for
          containing a query term. N and n count the *paper* rows only: the
          candidate universe is the document collection, and letting contexts
          into the count would make every score depend on how many contexts a
          split happens to carry.

  tokens  lowercase [a-z0-9]+, no stemming and no stopword list. Stopwords are
          left to idf, which is what it is for. `core.hardness` keeps its own
          stoplist for a different job -- deciding whether a title is present
          in a window -- and the two must not be merged.

  k1, b   1.5 and 0.75, the standard defaults, deliberately untuned. A tuned
          BM25 would be a stronger base; an untuned one is the more honest
          baseline and the weaker claim.

Scores are non-negative and a candidate sharing no query term scores exactly
0.0, so an all-miss pool ties and is resolved by the harness's tie rule rather
than by floating-point noise.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import polars as pl

TOKEN = __import__("re").compile(r"[a-z0-9]+")

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25Scorer:
    """BM25 over a `(key, kind, text)` parquet. Implements `PoolScorer`."""

    def __init__(self, texts: str | Path, *, k1: float = K1, b: float = B) -> None:
        frame = pl.read_parquet(texts, columns=["key", "kind", "text"])
        papers = frame.filter(pl.col("kind") == "paper")
        contexts = frame.filter(pl.col("kind") == "context")

        self.k1, self.b = k1, b
        self._queries = {
            k: tokenize(t)
            for k, t in zip(contexts["key"].to_list(), contexts["text"].to_list(), strict=True)
        }
        self._tf: dict[str, Counter] = {}
        self._len: dict[str, int] = {}
        df: Counter = Counter()
        for k, t in zip(papers["key"].to_list(), papers["text"].to_list(), strict=True):
            toks = tokenize(t)
            self._tf[k] = Counter(toks)
            self._len[k] = len(toks)
            # .keys(), not the Counter: `Counter.update(Counter)` ADDS the
            # frequencies, which would make df count occurrences rather than
            # documents. A term repeated 10 times in one document would then
            # report n=10 against N=2, driving idf negative and inverting the
            # pool's ordering.
            df.update(self._tf[k].keys())

        n_docs = len(self._tf)
        if n_docs == 0:
            raise SystemExit(
                f"{texts}: no rows with kind='paper'; BM25 needs the candidate "
                "documents to build its document frequencies from"
            )
        self._idf = {
            w: math.log(1.0 + (n_docs - n + 0.5) / (n + 0.5)) for w, n in df.items()
        }
        self._avgdl = sum(self._len.values()) / n_docs

    def _score(self, query: Sequence[str], candidate_id: str) -> float:
        try:
            tf, length = self._tf[candidate_id], self._len[candidate_id]
        except KeyError:
            raise KeyError(
                f"candidate {candidate_id!r} has no paper row in this text file"
            ) from None
        norm = self.k1 * (1.0 - self.b + self.b * length / self._avgdl)
        total = 0.0
        for word in query:
            freq = tf.get(word, 0)
            if freq:
                total += self._idf.get(word, 0.0) * freq * (self.k1 + 1.0) / (freq + norm)
        return total

    def _query(self, context_id: str) -> Sequence[str]:
        try:
            return self._queries[context_id]
        except KeyError:
            raise KeyError(
                f"context {context_id!r} has no context row in this text file"
            ) from None

    def __call__(self, context_id: str, candidate_id: str) -> float:
        return self._score(self._query(context_id), candidate_id)

    def pool_scores(
        self, context_id: str, candidates: Sequence[str]
    ) -> dict[str, float]:
        query = self._query(context_id)
        return {c: self._score(query, c) for c in candidates}
