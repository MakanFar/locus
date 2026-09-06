"""The repository must ship the licence its metadata promises.

`pyproject.toml` declared Apache-2.0 for two tags while shipping no LICENSE
file at all. These tests read the declaration rather than restating it, so a
future change of licence in one place fails here instead of shipping a repo
whose metadata and text disagree.
"""
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def test_the_declared_licence_has_a_file_naming_it():
    declared = _pyproject()["project"]["license"]["text"]
    text = (ROOT / "LICENSE").read_text()
    assert declared == "Apache-2.0"
    assert "Apache License" in text
    assert "Version 2.0" in text


def test_notice_states_that_no_upstream_text_is_redistributed():
    # This claim is the legal basis for the ID-only export: the identifiers
    # are ours to publish, the abstracts and context windows are not. If the
    # sentence goes, export.py's whole design rationale goes with it.
    notice = (ROOT / "NOTICE").read_text()
    assert "Gu" in notice, "NOTICE must credit the upstream dataset's authors"
    assert re.search(r"no upstream text", notice, re.IGNORECASE)


def test_changelog_has_an_entry_for_the_current_version():
    version = _pyproject()["project"]["version"]
    assert f"[{version}]" in (ROOT / "CHANGELOG.md").read_text()
