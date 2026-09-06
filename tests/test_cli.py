"""Every advertised verb must exist and be reachable.

The failure mode this guards is a README that documents a command nobody has
run -- which this repository has already shipped twice.
"""
import subprocess
import sys

import pytest

from locus.cli import VERBS, build_parser


def test_every_verb_parses():
    """Every key in `VERBS` must be a verb path `build_parser()` accepts.

    The brief this was written against asked for a `--help-probe` flag
    invented solely so the test could call `parse_args`. That would ship
    CLI surface that exists only for this test, which is worse than the
    problem it solves. Instead this calls `parse_known_args` -- the exact
    method `locus.cli.main` uses -- with no flags at all: leaf subparsers
    take no arguments of their own (they forward everything to the target
    callable, `--help` included), so a *valid* verb path parses cleanly
    with an empty remainder, and an invalid one raises `SystemExit` before
    this assertion is reached.
    """
    parser = build_parser()
    for verb in VERBS:
        args, remainder = parser.parse_known_args(verb.split())
        resolved = args.verb if not hasattr(args, "subverb") else f"{args.verb} {args.subverb}"
        assert resolved == verb
        assert remainder == []


@pytest.mark.parametrize("verb", sorted(VERBS))
def test_every_verb_resolves_to_a_real_callable(verb):
    """A verb whose target does not import is a broken command, and argparse
    will not tell you: it fails only when someone runs it."""
    fn = VERBS[verb]()
    assert callable(fn)


def test_help_imports_nothing_heavy():
    """`locus --help` must work with only the base install: torch,
    transformers, gensim and boto3 are all optional extras of this project,
    so importing `locus.cli` and building its parser must never reach for
    any of them.

    Run in a fresh interpreter, not this test's own process: another test
    module gated by e.g. `pytest.importorskip("gensim")` may already have
    imported one of these before this test runs, which would make a
    same-process `sys.modules` check meaningless.
    """
    code = (
        "import sys\n"
        "import locus.cli\n"
        "locus.cli.build_parser()\n"
        "bad = [m for m in ('torch', 'transformers', 'gensim', 'boto3') "
        "if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
