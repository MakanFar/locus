"""Interfaces the benchmark is defined in terms of.

Protocols live in `core` rather than beside their implementations so that
`scoring/` holds implementations only, and so `core` never imports upward to
state an interface.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Association(Protocol):
    """A symmetric association measure over paper ids.

    `PPMI` (cocitation.py) and `NodeVectors` (node2vec.py) already implement
    this exact surface; declaring it removes core's dependency on either and
    makes the polymorphism `association()` relies on checkable.
    """

    def value(self, c: str, a: str) -> float: ...

    def mean_over(self, c: str, anchors: frozenset[str]) -> float: ...

    def aggregate(self, c: str, anchors: frozenset[str], how: str = "mean") -> float: ...

    # The batch path `sweep`, `eval.rank` and `eval.swap` use to avoid
    # rebuilding a pool's candidate view per anchor. `prepare`'s return type is
    # implementation-private -- a CSR column index for PPMI, a dense matrix for
    # NodeVectors -- and is only ever handed straight back to `add_to`, so it
    # is deliberately opaque here rather than forced into a common shape.
    def prepare(self, candidates: list[str]) -> Any: ...

    def add_to(self, sums: Any, anchor: str, prep: Any) -> None: ...


@runtime_checkable
class Scorer(Protocol):
    """Assigns a real number to a (context, candidate) pair. Higher is better.

    Scorers are CALLABLES, not objects with a .score method: DenseScorer and
    MmapScorer implement __call__, and baselines.py's factories return plain
    closures. `baselines.Scorer` is the structural alias for the same thing.
    """

    def __call__(self, context_id: str, candidate_id: str) -> float: ...


@runtime_checkable
class PoolScorer(Scorer, Protocol):
    """A scorer that can score a whole pool at once.

    The sweep scores one pool at many lambdas, so going through __call__ per
    candidate would repeat the same dot products for every lambda.
    """

    def pool_scores(self, context_id: str, candidates: Sequence[str]) -> dict[str, float]: ...
