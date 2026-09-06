"""Which Hugging Face revision the local embedder pins, per model.

The SPECTER2 vectors on disk were produced at one pinned revision, and that
pin must not silently apply to every other model name: `--model
malteos/scincl` at SPECTER2's commit hash does not exist and fails inside
transformers with a message about the wrong repository.
"""
import pytest

from locus.scoring.embed import MODEL_REVISIONS, revision_for


def test_specter2_stays_pinned_to_the_published_revision():
    assert MODEL_REVISIONS["allenai/specter2_base"] == "3447645e"
    assert revision_for("allenai/specter2_base") == "3447645e"


def test_scincl_has_its_own_pin_not_specter2s():
    assert "malteos/scincl" in MODEL_REVISIONS
    assert revision_for("malteos/scincl") != "3447645e"


def test_an_unpinned_model_gets_no_revision_and_says_so(capsys):
    assert revision_for("some/other-encoder") is None
    assert "not pinned" in capsys.readouterr().err


def test_an_explicit_revision_overrides_the_pin():
    assert revision_for("allenai/specter2_base", "deadbeef") == "deadbeef"


def test_a_pin_must_look_like_a_commit_prefix():
    for rev in MODEL_REVISIONS.values():
        assert len(rev) >= 7 and all(c in "0123456789abcdef" for c in rev)
    with pytest.raises(ValueError, match="revision"):
        revision_for("allenai/specter2_base", "main branch")
