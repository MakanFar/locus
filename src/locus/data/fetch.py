"""Download a released artefact bundle and verify it against its manifest.

`urllib.request` from the stdlib, deliberately: a benchmark whose selling point
is reproducibility should not add a runtime dependency to download its own
files, and `file://` support is what lets this be tested with no network.

Each file is streamed to a temporary path and hashed WHILE streaming, then
moved into place only once the digest matches. A partial or corrupt artefact
that looks present is worse than an absent one: every downstream check would
run on truncated data and report something plausible.

The manifest is untrusted remote input naming both a local write target and a
remote read target, so both sides are validated -- on different
representations of the name, independently of each other:

  1. the URL actually resolved and opened (`urljoin(url, name)`) is checked,
     after percent-decoding, to still live in the manifest's own directory;
  2. the name itself, also decoded, is rejected if it is anything other than
     a single plain path component;
  3. the local write path built from that name is re-verified, after
     `Path.resolve()`, to sit directly inside `dest`.

No single one of these is trusted alone: a check on the raw (still-encoded)
name is not a check on the URL that gets opened, and a check on the URL is
not a check on the path that gets written -- percent-encoding is exactly an
encoding that can pass a check aimed at one representation while still being
decoded by the code that acts on another.

Three further things the confinement checks above do NOT cover, each handled
below:

  * **Redirects.** `urlopen` follows 3xx by default, and the confinement
    check above is performed on the URL as resolved, BEFORE it is opened -- so
    a host that passes the check and then answers `302 Location: <anywhere>`
    is fetched from anywhere. With `verify=True` that is harmless: the sha256
    is computed over whatever actually arrived, so a redirect can deliver the
    manifest's bytes or nothing. With `--no-verify` there is no such backstop
    at all. Redirects are therefore bounded (`MAX_REDIRECTS`) when hashing is
    on, and REFUSED outright when it is off. Refusing them unconditionally
    would have been simpler and is wrong: real deposit hosts (Zenodo among
    them) answer file requests with a redirect to object storage on another
    host, so a fetcher that cannot follow one cannot fetch the release.

  * **Size.** The manifest declares each file's `bytes`. That was validated
    into existence and then never used: the whole body was written before the
    digest was even computed, so a manifest naming a 40 KB file could stream
    a hundred gigabytes into `dest` and only then be told the hash was wrong.
    The declared size is now a hard cap enforced DURING the stream, and a
    short file is rejected as firmly as an over-long one.

  * **Partial bundles.** The per-file guarantee ("nothing appears at `dest`
    unless it hashes") does not compose into a per-bundle one: a bundle that
    failed at file 7 of 12 left six good files in `dest`, and `locus doctor`
    would then call the bundle "present" -- which is precisely the "partial
    artefact that looks present" this module exists to prevent, one level up.
    So every file lands in a staging directory inside `dest` and the whole
    set is moved into place only once every file has been fetched and
    verified; any failure removes the staging directory and leaves `dest`
    exactly as it was found.
"""
from __future__ import annotations

import hashlib
import json
import posixpath
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# The core bundle's manifest on Zenodo (record 22260046, doi:10.5281/zenodo.22260046,
# version 0.2.0). This must be the manifest FILE, not the record page: every
# other file is resolved by `urljoin` against it, so it has to sit in the
# same `files/` directory as the artefacts it names.
ZENODO_CORE_RECORD = "https://zenodo.org/records/22260046"
DEFAULT_MANIFEST_URL: str | None = f"{ZENODO_CORE_RECORD}/files/bundle_manifest.json"

CHUNK = 1 << 20
MAX_REDIRECTS = 5


def _validate_manifest(manifest: object) -> dict:
    """Check the manifest has the shape this module depends on.

    A truncated or corrupted manifest download is plausible without any
    malice, so this fails with the same clear `SystemExit` reporting as the
    rest of the module rather than a bare `KeyError`/`AttributeError` from
    indexing into whatever JSON happened to come back.
    """
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise SystemExit(
            'manifest is malformed: expected a JSON object with a "files" object'
        )
    files = manifest["files"]
    for name, meta in files.items():
        if not isinstance(meta, dict) or not isinstance(meta.get("sha256"), str):
            raise SystemExit(
                f"manifest is malformed: entry {name!r} has no string \"sha256\" field"
            )
        # `bytes` is required, not optional, because it is the only bound on
        # how much a remote host can make this process write before the
        # digest can say anything. `bool` is excluded explicitly: it is an
        # `int` subclass, and `True` would otherwise read as a one-byte cap.
        size = meta.get("bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise SystemExit(
                f"manifest is malformed: entry {name!r} has no non-negative "
                'integer "bytes" field, so its download has no size bound'
            )
    return files


class _BoundedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Bound redirects, or refuse them when nothing would catch an escape.

    The confinement checks run on the URL this module resolves and is about
    to open. A redirect replaces that URL after the check has passed, so it
    is outside their reach by construction. sha256 verification IS in reach
    of it -- it is computed on the bytes that arrive, from wherever they
    arrive -- which is why redirects are tolerable exactly when hashing is on.
    """

    max_redirections = MAX_REDIRECTS

    def __init__(self, allow: bool) -> None:
        self.allow = allow

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not self.allow:
            raise SystemExit(
                f"{req.full_url} answered a {code} redirect to {newurl}. The "
                "confinement check runs on the URL this fetch resolved, not "
                "on wherever a redirect points, and --no-verify means no "
                "sha256 would catch the difference -- so redirects are "
                "refused without verification. Drop --no-verify, or fetch "
                "the redirect target directly."
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _opener(*, allow_redirects: bool) -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_BoundedRedirectHandler(allow_redirects))


def _decode_and_check_name(name: str) -> str:
    """Percent-decode a manifest-supplied name and reject it if unsafe.

    Checked on the DECODED form, because the name guard exists to stop a
    manifest entry from naming a path outside `dest` -- and percent-encoding
    is exactly the kind of thing that can hide a `/`, a `\\`, or a leading
    `.` from a check performed on the raw string, only for it to reappear
    once something downstream decodes it.
    """
    if "\x00" in name:
        raise SystemExit(f"manifest names an unsafe path: {name!r}")
    decoded = urllib.parse.unquote(name)
    if (
        not decoded.strip()
        or "/" in decoded
        or "\\" in decoded
        or decoded.startswith(".")
        or "\x00" in decoded
    ):
        raise SystemExit(f"manifest names an unsafe path: {name!r}")
    return decoded


def _resolve_fetch_url(manifest_url: str, name: str) -> str:
    """Resolve the URL a manifest entry actually fetches, and confine it.

    This is the check that matters most: it validates the RESOURCE that is
    about to be opened, not the string that named it. `urljoin` is applied
    first, exactly as the download path does, then the resulting URL's path
    is percent-decoded and normalised and required to still sit in the same
    directory as the manifest itself -- so `../SECRET.txt`, encoded or not,
    cannot walk the fetch outside the bundle's own directory.
    """
    target = urllib.parse.urljoin(manifest_url, name)
    base_path = posixpath.normpath(
        urllib.parse.unquote(urllib.parse.urlparse(manifest_url).path)
    )
    target_path = posixpath.normpath(
        urllib.parse.unquote(urllib.parse.urlparse(target).path)
    )
    if posixpath.dirname(target_path) != posixpath.dirname(base_path):
        raise SystemExit(
            f"manifest names a path outside its own directory: {name!r}"
        )
    return target


def _safe_target(name: str, dest: Path) -> Path:
    """Resolve a (decoded, already name-checked) file name to a path inside
    `dest`, re-verifying containment after `resolve()` as a last, independent
    line of defence -- e.g. against pathlib's own behaviour of discarding the
    left side of a `/` join when the right side turns out to be absolute.
    """
    target = (dest / name).resolve()
    if target.parent != dest.resolve():
        raise SystemExit(f"manifest names an unsafe path: {name!r}")
    return target


def fetch(manifest_url: str | None, dest: Path | str, *, verify: bool = True) -> dict:
    url = manifest_url or DEFAULT_MANIFEST_URL
    if not url:
        raise SystemExit(
            "No manifest URL. The released bundle has not been deposited yet, "
            "so there is no default; pass the URL explicitly."
        )
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    opener = _opener(allow_redirects=verify)

    with opener.open(url) as fh:
        manifest = json.loads(fh.read().decode("utf-8"))

    files = _validate_manifest(manifest)

    # Staged inside `dest` rather than in the system temp directory, so the
    # final move is a rename on the same filesystem and cannot half-copy a
    # multi-gigabyte file into place.
    staging = Path(tempfile.mkdtemp(dir=dest, prefix=".locus-incoming-"))
    try:
        staged: list[tuple[Path, Path]] = []
        for name, meta in files.items():
            decoded_name = _decode_and_check_name(name)
            target = _resolve_fetch_url(url, name)
            final_path = _safe_target(decoded_name, dest)
            limit = meta["bytes"]
            tmp = staging / decoded_name
            digest = hashlib.sha256()
            written = 0
            try:
                with opener.open(target) as src, open(tmp, "wb") as out:
                    while chunk := src.read(CHUNK):
                        written += len(chunk)
                        if written > limit:
                            # Refused mid-stream, not after the fact: the
                            # point of the cap is to bound what a remote host
                            # can make this process write, and a check after
                            # the last chunk bounds nothing.
                            raise SystemExit(
                                f"{name}: the remote file is longer than the "
                                f"{limit} bytes the manifest declares; the "
                                "download was stopped and nothing was written"
                            )
                        digest.update(chunk)
                        out.write(chunk)
            except urllib.error.URLError as exc:
                raise SystemExit(
                    f"{name}: named in the manifest but could not be fetched "
                    f"from {target} ({exc})"
                ) from None
            if written != limit:
                raise SystemExit(
                    f"{name}: got {written} bytes but the manifest declares "
                    f"{limit}; nothing was written"
                )
            if verify and digest.hexdigest() != meta["sha256"]:
                raise SystemExit(
                    f"{name}: sha256 {digest.hexdigest()} does not match the "
                    f"manifest's {meta['sha256']}; nothing was written"
                )
            staged.append((tmp, final_path))

        # Every file is fetched and verified before ANY of them is visible at
        # `dest`. A bundle that fails at file 7 leaves no file 1-6 behind for
        # `locus doctor` to mistake for a complete artefact.
        for tmp, final_path in staged:
            tmp.replace(final_path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return manifest
