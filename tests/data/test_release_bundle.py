"""The bundle must carry no upstream text, and its manifest must verify.

NOTICE claims the released files carry identifiers, integers and booleans
only. That claim is what makes the bundle redistributable, so it is tested
rather than asserted: every string in every released row must look like an
identifier, never prose.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

# An id in this corpus is digits; a title or abstract has spaces in it.
IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]+$")


def test_the_bundler_refuses_a_missing_export(tmp_path, capsys):
    from locus.data import bundle

    code = bundle.main(["--work", str(tmp_path), "--out", str(tmp_path / "d")])
    assert code != 0
    assert "export" in capsys.readouterr().err.lower()


def test_the_bundler_refuses_to_bundle_an_excluded_file(tmp_path, capsys):
    """`EXCLUDED` must be reachable and must fire before anything is copied.

    `embed_inputs_*.parquet` never actually lives under `work/export/` in
    the real corpus -- it lives directly under `work/` and `CORE` never
    names it -- so this constructs the one shape of input that DOES reach
    the check: an `export/` directory that (however it got there) contains
    a forbidden file. If this test passes while the check in `bundle.py` is
    a no-op or runs after the copy, it is testing nothing.
    """
    from locus.data import bundle

    work = tmp_path / "work"
    export = work / "export"
    export.mkdir(parents=True)
    (export / "locus_rank_test.jsonl").write_text('{"context_id": "1"}\n')
    (export / "embed_inputs_test.parquet").write_bytes(b"upstream abstract text")
    (work / "cocitation_train.npz").write_bytes(b"not a real npz, just bytes")

    out = tmp_path / "out"
    code = bundle.main(["--work", str(work), "--out", str(out)])

    assert code != 0
    err = capsys.readouterr().err.lower()
    assert "embed_inputs" in err
    # Refused before any copy, not after: nothing forbidden -- and nothing
    # at all -- should have reached `dest`.
    dest = out / "locus-v0.2.0"
    assert not dest.exists() or not any(dest.rglob("embed_inputs*"))


def test_the_bundler_refuses_a_pre_existing_forbidden_file_in_dest(tmp_path, capsys):
    """A stale file already sitting in `dest` must never be blessed by the
    manifest of a later run.

    This is the actual shape of the bug this module exists to prevent: the
    manifest is built from `dest.rglob("*")`, not from `_source_files`, and
    a plain `dest.mkdir(exist_ok=True)` never cleans `dest` first. So the
    normal workflow -- run the bundler, then re-run it after an export
    change, into the same `--out`/`--version` -- is enough for anything
    already sitting in `dest` to be picked up, hashed and declared safe by
    `bundle_manifest.json`, whether or not it ever passed the `EXCLUDED`
    check. This reproduces exactly that: a real first run, then a forbidden
    file dropped into the now-existing `dest`, then a second run.
    """
    from locus.data import bundle

    work = tmp_path / "work"
    export = work / "export"
    export.mkdir(parents=True)
    (export / "locus_rank_test.jsonl").write_text('{"context_id": "1"}\n')
    (work / "cocitation_train.npz").write_bytes(b"not a real npz, just bytes")

    out = tmp_path / "out"
    first = bundle.main(["--work", str(work), "--out", str(out)])
    assert first == 0

    dest = out / "locus-v0.2.0"
    assert dest.exists()
    before = (dest / "bundle_manifest.json").read_text()

    # Drop upstream text into the pre-existing output directory -- the
    # scenario a stray export, a manual copy, or a partial previous run
    # could produce -- then re-run the bundler exactly as before.
    (dest / "embed_inputs_test.parquet").write_bytes(b"upstream abstract text")

    second = bundle.main(["--work", str(work), "--out", str(out)])

    assert second != 0
    # The manifest must not have been rewritten to bless the stray file:
    # either it is untouched, or if rewritten it still excludes it.
    after = (dest / "bundle_manifest.json").read_text()
    assert "embed_inputs" not in after
    if after != before:
        manifest = json.loads(after)
        assert not any("embed_inputs" in name for name in manifest["files"])


def _string_fields(row):
    for key, value in row.items():
        if isinstance(value, str):
            yield key, value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    yield key, item


@pytest.mark.equivalence
def test_no_released_row_carries_anything_but_identifiers(exported_bundle):
    for path in sorted(exported_bundle.glob("*.jsonl")):
        with open(path) as f:
            for n, line in enumerate(f, 1):
                for key, value in _string_fields(json.loads(line)):
                    assert IDENTIFIER.match(value), (
                        f"{path.name}:{n} field {key!r} is not an identifier: "
                        f"{value[:60]!r} -- this is upstream text and must not "
                        f"be released (see NOTICE)"
                    )


@pytest.mark.equivalence
def test_the_cocitation_graph_carries_no_upstream_text():
    """NOTICE claims `cocitation_train.npz`'s only string array (`vocab`)
    holds identifiers and every other array holds counts or indices. Unlike
    the `.jsonl` rows above, this file is binary and was previously
    untested -- glob("*.jsonl") in the test above never reaches it.
    """
    work = os.environ.get("LOCUS_WORK_DIR")
    if not work:
        pytest.skip("needs LOCUS_WORK_DIR")
    path = Path(work) / "cocitation_train.npz"
    if not path.exists():
        pytest.skip(f"no {path}")

    with np.load(path, allow_pickle=False) as z:
        assert "vocab" in z.files
        for name in z.files:
            arr = z[name]
            if name == "vocab":
                assert arr.dtype.kind == "U", (
                    f"vocab has dtype {arr.dtype}, expected a string array"
                )
                for value in arr:
                    assert IDENTIFIER.match(str(value)), (
                        f"vocab entry {value!r} is not an identifier -- this "
                        "is upstream text and must not be released "
                        "(see NOTICE)"
                    )
            else:
                assert arr.dtype.kind in "iu", (
                    f"{name} has dtype {arr.dtype}, expected an integer "
                    "array -- NOTICE claims only vocab carries strings"
                )
