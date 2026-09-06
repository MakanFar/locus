"""Assemble the deposit bundle and its verifying manifest.

The bundle is what `locus fetch` downloads. It carries the ID-only export and
the co-citation graph; it does NOT carry embed_inputs_*.parquet, which holds
upstream titles, abstracts and context windows (see NOTICE).

Precomputed embeddings are a separate, optional deposit: they are derived
from upstream abstracts, and keeping them out of the core means the core's
DOI survives if they are ever withdrawn.

`CORE` is an allowlist -- the only paths this module ever reads from `work/`
-- but it is not the only thing standing between upstream text and `dest`:
`EXCLUDED` is a second, independent check on every file `CORE` would
otherwise pull in, so that a future change widening `CORE`, or a stray
`embed_inputs_*` file landing inside `work/export/` some other way, is still
refused rather than silently bundled. It is checked against every source
file BEFORE anything is copied -- not after -- so a forbidden file is never
even briefly present in `dest`, let alone left there if the run is
interrupted between the check and a later one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

CORE = ("export", "cocitation_train.npz")
EXCLUDED = ("embed_inputs",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files(work: Path) -> list[Path]:
    """Every file `CORE` would pull in from `work/`, flattened.

    A `CORE` entry that is a directory contributes its immediate files (this
    bundle never nests a subdirectory -- see the flattening note in `main`);
    a `CORE` entry that is a file contributes itself.
    """
    found: list[Path] = []
    for name in CORE:
        src = work / name
        if src.is_dir():
            found.extend(sorted(p for p in src.iterdir() if p.is_file()))
        elif src.exists():
            found.append(src)
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="locus bundle", description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--version", default="0.2.0")
    args = ap.parse_args(argv)

    export = args.work / "export"
    if not export.is_dir():
        print(f"no export directory at {export}; run `locus build export` first",
              file=sys.stderr)
        return 1

    for name in CORE:
        if not (args.work / name).exists():
            print(f"missing {args.work / name}", file=sys.stderr)
            return 1

    sources = _source_files(args.work)

    # Checked over every source file before anything is written: refusing
    # AFTER a copy would mean the offending file already sat in `dest` by
    # the time this function said no.
    for src in sources:
        if any(bad in src.name for bad in EXCLUDED):
            print(f"refusing to bundle {src.name}: it carries upstream text",
                  file=sys.stderr)
            return 1

    dest = args.out / f"locus-v{args.version}"

    # `dest` must be empty (or absent) before anything is written into it.
    # `dest.rglob("*")` -- not `_source_files` -- is what the manifest below
    # is built from, so any file already sitting in `dest` before this run
    # would be picked up, hashed and blessed by `bundle_manifest.json`
    # whether or not it ever passed the `EXCLUDED` check above: a stray
    # `embed_inputs_*.parquet` left over from a manual copy, a previous
    # version's export, anything. Refusing a non-empty `dest` outright means
    # the manifest can only ever describe files this run itself copied from
    # `sources`, which is the only thing standing behind the "no upstream
    # text" claim on every fetch of this bundle.
    if dest.exists() and any(dest.iterdir()):
        print(f"refusing to bundle into {dest}: it already exists and is "
              "not empty -- remove it or choose a fresh --out/--version",
              file=sys.stderr)
        return 1
    dest.mkdir(parents=True, exist_ok=True)

    # Flattened, not `shutil.copytree`'d into an `export/` subdirectory:
    # `locus.data.fetch` deliberately refuses any manifest entry whose
    # (decoded) name is anything but a single plain path component -- a `/`
    # is rejected outright, on purpose, so that no manifest entry can ever
    # resolve outside `dest`. A nested `export/...` entry would be
    # unfetchable by the very fetcher this bundle exists to feed, so every
    # file this bundle carries lands directly inside `dest`, beside
    # `bundle_manifest.json`.
    for src in sources:
        shutil.copy2(src, dest / src.name)

    files: dict[str, dict] = {}
    for path in sorted(dest.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(dest))
        files[rel] = {"bytes": path.stat().st_size, "sha256": sha256(path)}

    manifest = {
        "schema": 1,
        "version": args.version,
        "license": "CC-BY-4.0",
        "notice": "Identifiers, integers and booleans only. No upstream text.",
        "files": files,
    }
    (dest / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(f["bytes"] for f in files.values())
    print(f"{dest}: {len(files)} files, {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
