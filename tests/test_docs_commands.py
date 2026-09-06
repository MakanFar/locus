"""Every `locus ...` command printed in the docs must be a real verb.

This repository has shipped a documented command that did not work, twice.
tests/test_cli.py forecloses that for the verb registry; this does it for
the documentation, which is where a reader actually finds the command.
"""
import re
from pathlib import Path

import pytest

from locus import cli

DOCS = Path(__file__).resolve().parents[1] / "docs"
INVOCATION = re.compile(r"locus ((?:[a-z][a-z-]*)(?: [a-z][a-z-]*)?)")


def verbs_in(path: Path) -> set[str]:
    """Every `locus <verb>` or `locus <verb> <sub>` printed in a document."""
    found = set()
    for match in INVOCATION.findall(path.read_text()):
        one, _, two = match.partition(" ")
        if two and f"{one} {two}" in cli.VERBS:
            found.add(f"{one} {two}")
        elif one in cli.VERBS:
            found.add(one)
        else:
            found.add(match)
    return found


@pytest.mark.parametrize("doc", sorted(DOCS.glob("*.md")), ids=lambda p: p.name)
def test_every_documented_locus_command_is_a_registered_verb(doc):
    unknown = {v for v in verbs_in(doc) if v not in cli.VERBS}
    assert not unknown, f"{doc.name} documents non-existent verbs: {sorted(unknown)}"


def test_custom_corpus_guide_names_the_fixture_it_is_checked_against():
    doc = (DOCS / "custom-corpus.md").read_text()
    assert "tests/data/fixtures/second_corpus" in doc


def test_scoring_guide_documents_the_real_protocol_surface():
    # protocols.py once described a `.score()` method that nothing
    # implemented. The real surface is __call__ plus pool_scores; the guide
    # must name what a user actually has to write.
    from locus.core.protocols import PoolScorer

    doc = (DOCS / "scoring.md").read_text()
    for name in ("Scorer", "PoolScorer"):
        assert name in doc
    for method in ("__call__", "pool_scores"):
        assert method in doc, f"scoring.md never mentions {method}"
    assert "score(" not in doc.replace("pool_scores(", ""), (
        "scoring.md describes a .score() method; the protocol has none"
    )

    # The class the guide tells a user to write must actually satisfy the
    # protocol. `__protocol_attrs__` would say this more directly but is
    # 3.12+, and this repository supports 3.11.
    class Documented:
        def __call__(self, context_id: str, candidate_id: str) -> float:
            return 0.0

        def pool_scores(self, context_id, candidates):
            return {c: self(context_id, c) for c in candidates}

    assert isinstance(Documented(), PoolScorer)
