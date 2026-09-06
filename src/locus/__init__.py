"""LOCUS: a within-paper citation-location discrimination benchmark.

`__version__` is read from the installed distribution's metadata rather than
written here, so it cannot drift from the version `pyproject.toml` declares.
An editable install is still an install, so this resolves in development too;
the fallback covers running straight out of a source tree that was never
installed at all.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("locus-benchmark")
except PackageNotFoundError:  # not installed: a bare source checkout
    __version__ = "0+unknown"

__all__ = ["__version__"]
