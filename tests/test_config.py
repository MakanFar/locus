"""LOCUS_DATA_DIR has no sensible default. A path from one laptop silently
resolves to nothing on any other machine, and the resulting error surfaces
thousands of lines later as an empty index."""
import importlib
import os
from pathlib import Path

import pytest


def test_missing_data_dir_fails_with_an_actionable_message(monkeypatch):
    monkeypatch.delenv("LOCUS_DATA_DIR", raising=False)
    import locus.config as config

    importlib.reload(config)
    with pytest.raises(SystemExit) as exc:
        config.data_dir()
    msg = str(exc.value)
    assert "LOCUS_DATA_DIR" in msg
    assert "contexts.json" in msg


def test_data_dir_returns_the_env_value(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCUS_DATA_DIR", str(tmp_path))
    import locus.config as config

    importlib.reload(config)
    assert config.data_dir() == tmp_path


def test_work_dir_default_is_not_package_relative(monkeypatch, tmp_path):
    """Regression test for the src/ move: WORK_DIR used to default to
    `Path(__file__).parent / "work"`, which was correct when `config.py`
    lived at `benchmark/locus/config.py` but now resolves to
    `src/locus/work` -- inside the installed package -- once this is
    pip-installed. A driver run without LOCUS_WORK_DIR set would then try to
    write frozen artefacts into site-packages.

    `config.WORK_DIR` is a module-scope constant computed at import time, so
    observing a different LOCUS_WORK_DIR requires `importlib.reload`, and
    that reload permanently overwrites the module global -- it does not get
    undone when monkeypatch restores the environment at teardown, only the
    next reload would re-pick-it-up. To avoid leaking a bad WORK_DIR into
    whichever test runs after this one, we save the pre-test env value
    ourselves and restore it *and* reload `config` again in a `finally`
    block, rather than relying on monkeypatch's teardown timing relative to
    our own reload.
    """
    import locus.config as config

    original_env = os.environ.get("LOCUS_WORK_DIR")
    try:
        # Unset: the default must not resolve inside the installed package.
        monkeypatch.delenv("LOCUS_WORK_DIR", raising=False)
        importlib.reload(config)
        package_dir = Path(config.__file__).resolve().parent
        assert package_dir != config.WORK_DIR.resolve()
        assert package_dir not in config.WORK_DIR.resolve().parents

        # Set: WORK_DIR must be exactly the configured path.
        monkeypatch.setenv("LOCUS_WORK_DIR", str(tmp_path))
        importlib.reload(config)
        assert tmp_path == config.WORK_DIR
    finally:
        if original_env is None:
            monkeypatch.delenv("LOCUS_WORK_DIR", raising=False)
        else:
            monkeypatch.setenv("LOCUS_WORK_DIR", original_env)
        importlib.reload(config)
