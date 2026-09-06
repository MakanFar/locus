# tests/equivalence/test_frozen_config.py
"""The frozen constants are published numbers, not tuning knobs.

The chance floor is H_k/k and it MOVES with k -- 0.2928968253968254 at k=10,
0.3143... at k=9. That value appears in the paper, in the README, in every
exported manifest and in `golden_digest.json`, and consumers check it by
equality. So the constant, the artefacts on disk and the published floor have
to agree, and nothing else in the suite makes them: the digest is generated
from the exported pools, and the export derives k from those pools by design
(`pool_size_of`) precisely so the manifest describes the file rather than our
configuration. Editing `config.POOL_SIZE` alone therefore moved no test at
all before this file existed.

That was the argument for pinning POOL_SIZE. It applies just as well to the
other five, and for a while this file pinned only the one: `SWAP_K` 5 -> 3,
`HARDNESS_TAU`, `THETA` and `SHINGLE_N` each left the whole suite green AND
this gate green, because the assertion below read one key out of a dict that
already carried all six. Only `SEED` was caught, and by a different test.

So the check is driven BY the manifest rather than by a list written here:
every key `data.export` puts in `manifest["config"]` is looked up as the
upper-cased attribute of the same name on `locus.config` and required to
match. Freezing a seventh constant into the manifest brings it under this
gate with no edit to this file -- which is the property that was missing, not
the five individual assertions.
"""
import json

import pytest

from locus import config
from locus.data.export import rank_floor

pytestmark = pytest.mark.equivalence

# The floor as published. This literal is the paper's, and it is why the
# assertions below are equality rather than approximate.
PUBLISHED_RANK_FLOOR = 0.2928968253968254


def test_chance_floor_is_the_published_constant():
    """H_{POOL_SIZE}/POOL_SIZE, bit-exactly. Needs no artefacts, so it holds
    on a bare clone as well as under the full gate."""
    assert rank_floor(config.POOL_SIZE) == PUBLISHED_RANK_FLOOR


def _manifest(work_dir):
    return json.loads(
        (work_dir / "export" / "locus_manifest_test.json").read_text(
            encoding="utf-8"
        )
    )


def test_every_frozen_constant_agrees_with_the_exported_manifest(work_dir):
    """Each constant vs the artefacts that were built under it.

    Manifest-driven on purpose: iterating `manifest["config"]` means a
    constant cannot be frozen into the export and left unguarded here, which
    is exactly how five of the six escaped. `config` also carries constants
    that are NOT in the manifest (the gate thresholds); those are pinned
    separately below, against the README, since no artefact records them.
    """
    frozen = _manifest(work_dir)["config"]
    assert frozen, "manifest carries no frozen config to check"
    for key, value in sorted(frozen.items()):
        attr = key.upper()
        assert hasattr(config, attr), (
            f"manifest freezes {key!r} but locus.config has no {attr}; a "
            "constant in the export with no constant behind it cannot drift "
            "detectably in either direction"
        )
        assert value == getattr(config, attr), (
            f"config.{attr} is {getattr(config, attr)!r} but the exported "
            f"artefacts were built with {key}={value!r}"
        )


def test_pool_size_agrees_with_the_exported_manifest(work_dir):
    """The manifest records the pool size twice -- once as the frozen config
    it was built under, once as the k of the pools actually written -- and
    both are checked, because a mismatch between those two is its own kind of
    bug."""
    manifest = _manifest(work_dir)
    assert manifest["config"]["pool_size"] == config.POOL_SIZE
    assert manifest["floors"]["pool_size"] == config.POOL_SIZE
    assert manifest["floors"]["rank_constant_mrr"] == repr(PUBLISHED_RANK_FLOOR)


def test_gate_thresholds_match_the_published_table():
    """README.md's *Gates* table, pinned.

    These four were bare literals inside `build/pipeline.py` until they moved
    to `config`, and editing any of them moved no test either -- the same hole
    the six constants above had, on the numbers a reviewer reads out of the
    README to decide whether their own build passed. Needs no artefacts.
    """
    assert config.MAX_ALIGNMENT_FAILURE == 0.15
    assert config.MIN_ANCHOR_COVERAGE == 0.60
    assert config.HARD_FRACTION_BAND == (0.40, 0.85)
    assert config.MIN_HEADLINE == 30_000
