"""The protocol must describe the implementations that already exist, not an
aspiration. If PPMI and NodeVectors ever diverge, this fails.

Caveat: `runtime_checkable` only checks that the named methods exist, not
their signatures -- an `isinstance` check here would pass an object with a
`pool_scores` attribute of the wrong arity or return type. What it does
catch, loudly, is the drift that actually happens: a method getting renamed
or removed out from under the protocol.
"""
from locus.core.protocols import Association, PoolScorer, Scorer
from locus.scoring.baselines import constant_scorer, popularity_scorer


def test_ppmi_satisfies_association(tiny_graph):
    assert isinstance(tiny_graph.ppmi(alpha=0.75, min_count=1), Association)


def test_raw_counts_satisfies_association(tiny_graph):
    assert isinstance(tiny_graph.raw_counts(min_count=1), Association)


def test_node_vectors_satisfies_association(tiny_vectors):
    assert isinstance(tiny_vectors, Association)


def test_dense_scorer_satisfies_scorer_and_pool_scorer(dense_scorer):
    assert isinstance(dense_scorer, Scorer)
    assert isinstance(dense_scorer, PoolScorer)


def test_popularity_scorer_satisfies_scorer():
    assert isinstance(popularity_scorer({"a": 3}), Scorer)


def test_constant_scorer_satisfies_scorer():
    assert isinstance(constant_scorer(1.0), Scorer)


def test_a_plain_closure_scorer_is_not_a_pool_scorer():
    # The real distinction PoolScorer draws: baselines.py's factories return
    # plain closures with no pool_scores method at all, so a caller that
    # needs the batch fast path (the lambda sweep) can tell at runtime that
    # this scorer does not offer it and must fall back to __call__ per
    # candidate, rather than assuming every Scorer is also a PoolScorer.
    assert isinstance(constant_scorer(1.0), Scorer)
    assert not isinstance(constant_scorer(1.0), PoolScorer)
