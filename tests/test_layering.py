"""Dependency direction is downward and enforced, not merely intended.

experiments -> eval -> build -> {scoring, data} -> core

`scoring` and `data` share a rank: both may import `core` and each other. Any
edge from a lower rank to a higher one is a design error, because it means a
layer cannot be read, tested or reused without the thing above it.

**Known limitation: only static imports are seen.** This walks the AST for
`import` and `from ... import` statements. A module fetched at run time through
`importlib.import_module`, `__import__` or an `exec` of a constructed name is
invisible here and would cross a layer boundary undetected. `src/` contains no
such call today; if one is ever added, this test cannot be the thing that
guards it.
"""
import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "locus"

LAYERS = {
    "core": 0,
    "data": 1,
    "scoring": 1,
    "build": 2,
    "eval": 3,
    "experiments": 4,
}

# Modules that sit directly in src/locus/ rather than in a layer package.
# Both maps are explicit and exhaustive: an unrecognised top-level module
# fails the test rather than being skipped, because the previous early return
# ("top-level modules: config.py, cli.py, __init__.py") silently exempted the
# 252-line embed_specter2.py as well, and an exemption nobody chose is not an
# exemption.
TOP_LEVEL_LAYERS = {
    # An artefact producer: it writes the embeddings parquet that scoring
    # consumes, so it belongs at build's rank and is held to build's limits
    # even though it lives outside build/. It imports nothing from locus
    # today; the point of naming it is that it cannot start importing eval/
    # or experiments/ without this failing.
    "embed_specter2.py": "build",
}
UNLAYERED = {
    # Rank 0 by construction and depended on from everywhere: config.py holds
    # paths and frozen constants and imports nothing from locus, and the
    # package __init__ carries only __version__.
    "config.py",
    "__init__.py",
    # The composition root: cli.py's whole job is to dispatch to every
    # layer, from data/scoring through build and eval up to experiments, so
    # by design its resolvers import all of them -- this walk sees those
    # imports even though each one sits inside a function body and is not
    # reached until that verb actually runs (the module-level docstring
    # explains why). Assigning it a layer would have to be "experiments"
    # (rank 4, the highest one that exists) to avoid a false failure, and
    # that check would then be permanently vacuous: nothing outranks
    # "experiments" today, so no import cli.py could ever make would be
    # flagged. That is indistinguishable from not checking it at all, so
    # it is named here instead of given a rank that implies a restriction
    # this module was never meant to have.
    "cli.py",
}

ALLOWED_UPWARD = {
    # data/export.py's CLI verifies what it just wrote. The dependency is on
    # the command line, not on the module: nothing in data/ imports eval/ at
    # module scope. Any addition here needs the same justification.
    #
    # Keyed on the path relative to src/locus, not on the bare filename: as
    # `("export.py", "eval")` this exemption attached to any file named
    # export.py anywhere in the tree.
    ("data/export.py", "eval"),
}


def _rel(path: pathlib.Path) -> str:
    return path.relative_to(SRC).as_posix()


def _layer_of(path: pathlib.Path) -> str | None:
    rel = path.relative_to(SRC).parts
    if len(rel) == 1:
        if rel[0] in TOP_LEVEL_LAYERS:
            return TOP_LEVEL_LAYERS[rel[0]]
        assert rel[0] in UNLAYERED, (
            f"{rel[0]} sits directly in src/locus/ and is in neither "
            "TOP_LEVEL_LAYERS nor UNLAYERED. Give it a layer, or name it as "
            "deliberately unlayered -- do not leave it unclassified, because "
            "unclassified means unchecked."
        )
        return None
    return rel[0] if rel[0] in LAYERS else None


def _own_package(path: pathlib.Path) -> str:
    """The package a module's relative imports resolve against.

    For src/locus/eval/verify.py that is `locus.eval`; for a package's own
    `__init__.py` it is that package itself, which is what dropping the last
    path component gives in both cases.
    """
    return ".".join(("locus", *path.relative_to(SRC).parts[:-1]))


def _absolute(node: ast.ImportFrom, path: pathlib.Path) -> str | None:
    """`from ..eval.verify import x` -> `locus.eval.verify`.

    `node.level` is the count of leading dots and was previously never read,
    so every relative import was invisible: `node.module` is `"eval"` there,
    which does not start with `"locus."`, and for `from .. import eval` it is
    None. Level 1 means the containing package, level 2 its parent, and so on.
    """
    if node.level == 0:
        return node.module
    parts = _own_package(path).split(".")
    base = parts[: len(parts) - (node.level - 1)]
    if not base:
        return None  # more dots than package depth: not a locus import
    return ".".join(base + ([node.module] if node.module else []))


def _layer_of_dotted(name: str | None) -> str | None:
    if not name:
        return None
    parts = name.split(".")
    if parts[0] != "locus" or len(parts) < 2:
        return None
    return parts[1] if parts[1] in LAYERS else None


def _imported_layers(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(_layer_of_dotted(alias.name))
        elif isinstance(node, ast.ImportFrom):
            base = _absolute(node, path)
            found.add(_layer_of_dotted(base))
            # `from locus import eval` and `from .. import eval` name the
            # submodule in the alias, not in `node.module`.
            for alias in node.names:
                found.add(_layer_of_dotted(f"{base}.{alias.name}" if base else None))
    return found - {None}


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=_rel)
def test_no_upward_imports(path):
    layer = _layer_of(path)
    if layer is None:
        return
    mine = LAYERS[layer]
    offenders = {
        other for other in _imported_layers(path)
        if LAYERS[other] > mine and (_rel(path), other) not in ALLOWED_UPWARD
    }
    assert not offenders, (
        f"{_rel(path)} is in '{layer}' (rank {mine}) but imports "
        f"{sorted(offenders)} from a higher rank"
    )
