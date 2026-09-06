# tests/equivalence/test_headline.py
"""The published headline number, recomputed end to end from the export.

`golden_digest.json` covers the numeric primitives -- aggregation, the
bootstrap, PPMI, the rescorer, the null controls -- but it never calls the
dense scorer and never runs the pipeline that turns those primitives into the
number the paper leads with. So the whole path from `locus_rank_*.jsonl` and
the embeddings parquet through `scoring.dense`, `core.probe`, `core.rescore`
and `eval.verify` could move without a single digest key changing.

This test closes that gap by calling `locus.eval.verify.primary_result`
directly (no subprocess: the function returns the floats, where the CLI only
prints them) and asserting the published values. It runs in about five seconds
on the frozen artefacts.

The assertions are on the 5-decimal rendering the module itself reports, via
`verify.format_row`. That is deliberate: 5 dp is the precision the paper
quotes, and it is the precision this file is allowed to claim. Asserting on
raw `repr` floats would be inventing digits that have never been published.
"""
import pytest

from locus.eval import verify

pytestmark = pytest.mark.equivalence

# From the module docstring of locus.eval.verify and from rank_report.json:
# test split, SPECTER2, alpha=0.5, min_count=1, lambda=4.0.
EXPECTED = {
    "MRR": "MRR      0.47771   0.59585  +0.11814   [+0.11062, +0.12601]",
    "R@1": "R@1      0.28051   0.43124  +0.15072   [+0.14051, +0.16115]",
}


@pytest.fixture(scope="session")
def headline_metrics(work_dir):
    """Computed once: this is the only slow thing in the equivalence gate."""
    embeddings = work_dir / "embeddings_test.parquet"
    graph = work_dir / "cocitation_train.npz"
    for path in (embeddings, graph):
        if not path.exists():
            pytest.skip(f"{path} is not present")
    return verify.primary_result(
        export=work_dir / "export",
        embeddings=embeddings,
        graph=graph,
        split="test",
        alpha=0.5,
        min_count=1,
        lam=4.0,
    )


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_primary_result_matches_the_published_row(name, headline_metrics):
    assert verify.format_row(name, headline_metrics[name]) == EXPECTED[name]


def test_delta_is_the_difference_of_the_two_reported_means(headline_metrics):
    """The delta is not an independently reported number: if it ever stops
    being `rescored - base`, the row above would still pass while the paper's
    arithmetic had come apart."""
    for m in headline_metrics.values():
        assert m.delta == m.rescored - m.base


def test_the_headline_interval_excludes_zero(headline_metrics):
    """The claim is a gain, not a difference. A CI straddling zero would be a
    different paper, and printing it at 5 dp would not necessarily show it."""
    for m in headline_metrics.values():
        assert m.ci_lo > 0.0
        assert m.ci_lo <= m.delta <= m.ci_hi
