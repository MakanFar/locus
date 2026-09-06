"""The train-only co-citation graph: counts and marginals, never PPMI.

A *site* is one (citing paper, location). It contributes the unordered pairs
of G(l) -- the refids observed as targets at that location -- once per site,
which is the co-citation event the paper defines. Marker anchors are deliberately
not edges: A(l) is the union of both routes at *scoring* time, but the graph is
built from observed co-citations only.

Endpoints are interned to int32 and accumulated in array.array buffers, then
collapsed with one np.unique. A dict[(str, str), int] over the ~2.6M distinct
train pairs would cost several hundred MB and reintroduce exactly the problem
this module exists to avoid.

PPMI is never frozen: alpha and the min-count are swept on val, and freezing
derived values would force a rebuild per sweep point.
"""
import array
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from locus.core.types import ContextRec, gold_sets


@dataclass(frozen=True)
class CoGraph:
    counts: sparse.csr_matrix
    marginals: np.ndarray
    n_total: int
    vocab: tuple[str, ...]
    index: dict[str, int]

    def save(self, path: str | Path) -> None:
        m = self.counts
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            indptr=m.indptr,
            indices=m.indices,
            data=m.data,
            marginals=self.marginals,
            n_total=np.int64(self.n_total),
            vocab=np.array(self.vocab),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CoGraph":
        with np.load(path, allow_pickle=False) as z:
            vocab = tuple(str(v) for v in z["vocab"])
            n = len(vocab)
            counts = sparse.csr_matrix(
                (z["data"], z["indices"], z["indptr"]), shape=(n, n)
            )
            return cls(
                counts=counts,
                marginals=z["marginals"],
                n_total=int(z["n_total"]),
                vocab=vocab,
                index={r: i for i, r in enumerate(vocab)},
            )

    def ppmi(self, alpha: float = 0.75, min_count: int = 2) -> "PPMI":
        coo = self.counts.tocoo()
        keep = coo.data >= min_count
        r, c = coo.row[keep], coo.col[keep]
        v = coo.data[keep].astype(np.float64)
        marg = self.marginals.astype(np.float64)
        # Context-distribution smoothing applied to BOTH endpoints:
        # P_alpha(x) = C(x)^alpha / Z, Z = sum_x C(x)^alpha. The graph is
        # undirected (C(c,a) = C(a,c)), so both roles must be smoothed the
        # same way -- smoothing only one side (as a naive extension of the
        # textbook PMI formula would suggest) makes the matrix asymmetric
        # for alpha != 1, which the co-citation graph itself is not. At
        # alpha=1, Z == n_total exactly, so Z**2/n_total collapses to
        # n_total and this is the textbook PMI closed form exactly.
        z = float((marg**alpha).sum())
        val = np.log(
            v * z * z / (self.n_total * (marg[r] ** alpha) * (marg[c] ** alpha))
        )
        pos = val > 0
        mat = sparse.csr_matrix(
            (val[pos], (r[pos], c[pos])), shape=self.counts.shape
        )
        mat.sort_indices()
        return PPMI(mat, self.index)

    def raw_counts(self, min_count: int = 1) -> "PPMI":
        """The co-count matrix itself, as the association measure.

        The "PPMI vs raw co-count" ablation: is the result robust to the
        association measure, or is it the PMI transform doing the work? The
        return type is deliberately the same PPMI view so every caller
        downstream is untouched -- only the numbers in the matrix differ.

        Counts and PPMI live on wildly different scales (counts run to the
        thousands, PPMI to about 12), so a lambda selected for one is
        meaningless for the other. This must be swept on val, never
        transplanted.
        """
        coo = self.counts.tocoo()
        keep = coo.data >= min_count
        mat = sparse.csr_matrix(
            (coo.data[keep].astype(np.float64), (coo.row[keep], coo.col[keep])),
            shape=self.counts.shape,
        )
        mat.sort_indices()
        return PPMI(mat, self.index)


@dataclass(frozen=True)
class PPMI:
    """A PPMI view of a CoGraph at one (alpha, min_count).

    Bundles the matrix with its vocabulary index so a caller cannot pair a
    matrix with the wrong index.
    """

    matrix: sparse.csr_matrix
    index: dict[str, int]

    def value(self, c: str, a: str) -> float:
        i, j = self.index.get(c), self.index.get(a)
        if i is None or j is None:
            return 0.0
        return float(self.matrix[i, j])

    def aggregate(self, c: str, anchors: frozenset[str],
                  how: str = "mean") -> float:
        """Pool the anchor set's evidence for candidate `c`.

        `mean` is the pre-registered aggregator and the one every headline
        number uses. The other two exist because the leave-one-out analysis
        makes a specific prediction about the mean's failure mode: an anchor
        with PPMI(c,a) = 0 contributes nothing to the numerator but still
        counts in the denominator, so it dilutes whatever evidence the other
        anchors supply and its removal *improves* the target's rank. That is
        a property of the aggregator, not of the citation.

          mean         total / |A|
          masked_mean  total / |{a : PPMI(c,a) > 0}|, and 0.0 if that set is
                       empty -- removes the dilution pathway exactly
          max          max_a PPMI(c,a) -- ignores set size altogether

        The three differ in scale by roughly an order of magnitude, so lambda
        is re-swept on val for each; comparing them at one lambda would
        measure the scale mismatch rather than the aggregator.
        """
        if how not in ("mean", "masked_mean", "max"):
            raise ValueError(f"unknown aggregator {how!r}")
        if not anchors:
            return 0.0
        i = self.index.get(c)
        if i is None:
            return 0.0
        lo, hi = self.matrix.indptr[i], self.matrix.indptr[i + 1]
        row = dict(
            zip(
                self.matrix.indices[lo:hi].tolist(),
                self.matrix.data[lo:hi].tolist(),
                strict=True,
            )
        )
        cols = sorted(j for a in anchors if (j := self.index.get(a)) is not None)
        vals = [v for j in cols if (v := row.get(j, 0.0)) > 0.0]
        if how == "max":
            return max(vals) if vals else 0.0
        if how == "masked_mean":
            return math.fsum(vals) / len(vals) if vals else 0.0
        # An anchor outside the train vocabulary is evidence we do not have,
        # not evidence that does not exist, so the mean divides by the full
        # |A| rather than by the number of anchors that reached the graph.
        return math.fsum(vals) / len(anchors)

    def prepare(self, candidates: list[str]) -> dict[int, list[int]]:
        """Per-location index for repeated anchor scatter-adds.

        Sequential completion adds one anchor at a time to a fixed candidate list and
        rescores after each, so the candidate-to-graph-column map is built
        once per location rather than once per anchor.
        """
        pos_of_col: dict[int, list[int]] = {}
        for i, c in enumerate(candidates):
            j = self.index.get(c)
            if j is not None:
                pos_of_col.setdefault(j, []).append(i)
        return pos_of_col

    def add_to(self, sums, anchor: str, prep) -> None:
        """Accumulate this anchor's association with every candidate.

        Walks the anchor's CSR row and scatters into the candidates that share
        a column. An anchor outside the vocabulary contributes nothing, which
        is the same treatment `aggregate` gives it.
        """
        j = self.index.get(anchor)
        if j is None:
            return
        lo, hi = self.matrix.indptr[j], self.matrix.indptr[j + 1]
        cols = self.matrix.indices[lo:hi].tolist()
        vals = self.matrix.data[lo:hi].tolist()
        for col, val in zip(cols, vals, strict=True):
            for i in prep.get(col, ()):
                sums[i] += val

    def mean_over(self, c: str, anchors: frozenset[str]) -> float:
        """(1/|A|) * sum_{a in A} PPMI(c,a). Exactly 0.0 for an empty A.

        Summed via math.fsum over anchors sorted by column index rather than
        accumulated with `+=` in frozenset iteration order: frozenset[str]
        order depends on PYTHONHASHSEED and on the order the set was built
        in, and float addition is not associative, so the naive approach
        gave the same (candidate, anchors, PPMI) triple different last-ULP
        values across processes.
        """
        if not anchors:
            return 0.0
        i = self.index.get(c)
        if i is None:
            return 0.0
        lo, hi = self.matrix.indptr[i], self.matrix.indptr[i + 1]
        row = dict(
            zip(
                self.matrix.indices[lo:hi].tolist(),
                self.matrix.data[lo:hi].tolist(),
                strict=True,
            )
        )
        cols = sorted(j for a in anchors if (j := self.index.get(a)) is not None)
        total = math.fsum(row.get(j, 0.0) for j in cols)
        # Divide by the full |A|: an anchor outside the train vocabulary is
        # evidence we do not have, not evidence that does not exist.
        return total / len(anchors)


def count_sites(
    papers: Iterable[tuple[str, list[ContextRec]]],
) -> CoGraph:
    seen: dict[str, int] = {}
    rows = array.array("i")
    cols = array.array("i")
    if rows.itemsize != 4:
        # np.frombuffer(..., np.int32) below assumes 4-byte elements; an
        # assert here would be stripped under python -O, silently letting a
        # platform where 'i' is not 4 bytes misread the buffer. Checked here,
        # right after the buffers are created, rather than after a whole
        # streaming pass has filled them with ~44 MB of data.
        raise SystemExit(
            f"array('i') is {rows.itemsize} bytes on this platform, not the "
            "4 bytes np.frombuffer(..., np.int32) below assumes"
        )

    for _citing_id, recs in papers:
        for refids in gold_sets(recs).values():
            ids = sorted(seen.setdefault(r, len(seen)) for r in refids)
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    rows.append(ids[a])
                    cols.append(ids[b])
                    rows.append(ids[b])
                    cols.append(ids[a])

    n = len(seen)
    vocab = tuple(sorted(seen))
    index = {r: i for i, r in enumerate(vocab)}
    if n == 0 or len(rows) == 0:
        empty = sparse.csr_matrix((n, n), dtype=np.int32)
        return CoGraph(empty, np.zeros(n, dtype=np.int64), 0, vocab, index)

    # Remap first-appearance ids onto the sorted vocabulary, so the frozen ids
    # do not depend on the order papers happened to arrive in.
    perm = np.empty(n, dtype=np.int32)
    for refid, old in seen.items():
        perm[old] = index[refid]
    r = perm[np.frombuffer(rows, dtype=np.int32)]
    c = perm[np.frombuffer(cols, dtype=np.int32)]

    key = r.astype(np.int64) * n + c
    uniq, counts = np.unique(key, return_counts=True)
    mat = sparse.csr_matrix(
        (
            counts.astype(np.int32),
            ((uniq // n).astype(np.int32), (uniq % n).astype(np.int32)),
        ),
        shape=(n, n),
    )
    mat.sort_indices()
    marginals = np.asarray(mat.sum(axis=1)).ravel().astype(np.int64)
    return CoGraph(mat, marginals, int(marginals.sum()), vocab, index)
