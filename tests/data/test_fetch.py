"""Fetching a release bundle, verified against its own manifest.

Tested over file:// so the code path runs with no network and no DOI -- the
Zenodo deposit does not exist yet. The property that matters is not that a
download works, but that a CORRUPT download is caught and leaves nothing
behind: a half-written artefact that looks present is worse than an absent one,
because every downstream check would then be running on truncated data.
"""
import hashlib
import http.server
import json
import socketserver
import threading

import pytest

from locus.data import fetch as fetch_mod
from locus.data.fetch import _resolve_fetch_url, fetch


def _publish(root, files):
    """Write payload files plus a manifest naming their real sha256s."""
    root.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name, body in files.items():
        (root / name).write_bytes(body)
        entries[name] = {
            "sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
        }
    (root / "manifest.json").write_text(json.dumps({"files": entries}))
    return (root / "manifest.json").as_uri()


def test_fetch_downloads_and_verifies(tmp_path):
    url = _publish(tmp_path / "src", {"a.jsonl": b"one\ntwo\n", "b.npz": b"\x00\x01"})
    dest = tmp_path / "dest"
    got = fetch(url, dest)
    assert (dest / "a.jsonl").read_bytes() == b"one\ntwo\n"
    assert (dest / "b.npz").read_bytes() == b"\x00\x01"
    assert set(got["files"]) == {"a.jsonl", "b.npz"}


def test_a_corrupt_payload_raises_and_leaves_no_file(tmp_path):
    """The manifest is the authority. If a byte differs, the file must not
    appear at the destination at all -- not even truncated."""
    src = tmp_path / "src"
    url = _publish(src, {"a.jsonl": b"correct payload"})
    (src / "a.jsonl").write_bytes(b"corrupted!!!!!!")  # same length, wrong bytes
    dest = tmp_path / "dest"
    with pytest.raises(SystemExit, match="sha256"):
        fetch(url, dest)
    assert not (dest / "a.jsonl").exists(), "a failed fetch left a partial file"


def test_a_file_named_in_the_manifest_but_absent_is_reported(tmp_path):
    src = tmp_path / "src"
    url = _publish(src, {"a.jsonl": b"x"})
    (src / "a.jsonl").unlink()
    dest = tmp_path / "dest"
    with pytest.raises(SystemExit) as exc:
        fetch(url, dest)
    assert "a.jsonl" in str(exc.value)


def test_verify_false_skips_hashing_but_still_downloads(tmp_path):
    src = tmp_path / "src"
    url = _publish(src, {"a.jsonl": b"correct"})
    (src / "a.jsonl").write_bytes(b"changed")
    dest = tmp_path / "dest"
    fetch(url, dest, verify=False)
    assert (dest / "a.jsonl").read_bytes() == b"changed"


def _publish_raw(root, manifest_text):
    """Write a manifest verbatim, without deriving it from real payloads --
    for manifest shapes and names that must be rejected before any payload
    fetch is attempted."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(manifest_text)
    return (root / "manifest.json").as_uri()


def test_percent_encoded_traversal_cannot_read_outside_the_bundle_dir(tmp_path):
    """`%2e%2e%2f` decodes to `../` only once the URL is opened -- a check on
    the raw manifest string alone would never see it. With verify=False no
    hash is even known, so an unguarded fetch here is an arbitrary local-file
    read: this pins that the fetched URL itself, not just the name, is
    confined to the manifest's own directory."""
    secret = tmp_path / "SECRET.txt"
    secret.write_bytes(b"topsecret")
    src = tmp_path / "src"
    manifest = json.dumps(
        {"files": {"%2e%2e%2fSECRET.txt": {"sha256": "0" * 64, "bytes": 9}}}
    )
    url = _publish_raw(src, manifest)
    dest = tmp_path / "dest"
    with pytest.raises(SystemExit, match="unsafe path"):
        fetch(url, dest, verify=False)
    assert not any(dest.glob("*")), "percent-encoded traversal leaked a file"


@pytest.mark.parametrize(
    "name",
    [
        "../escape.txt",
        "..",
        "/etc/passwd",
        "sub\\evil.txt",
        "..\\evil.txt",
        "",
        ".hidden",
        "%2e%2e%2fescape.txt",
        "%2e%2e%2f%2e%2e%2fetc%2fhostname",
        "%2f%2fetc%2fpasswd",
    ],
)
def test_unsafe_manifest_names_are_rejected_with_no_leak(tmp_path, name):
    (tmp_path / "src" / "escape.txt").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "escape.txt").write_bytes(b"outside the bundle dir")
    src = tmp_path / "src"
    manifest = json.dumps({"files": {name: {"sha256": "0" * 64, "bytes": 1}}})
    url = _publish_raw(src, manifest)
    dest = tmp_path / "dest"
    with pytest.raises(SystemExit):
        fetch(url, dest, verify=False)
    assert not any(dest.glob("*")), f"{name!r} leaked a file into dest"


def test_double_percent_encoded_traversal_does_not_escape(tmp_path):
    """A double-encoded `%252e%252e%252f` decodes ONCE (matching what the
    fetch layer itself decodes) to the literal string `%2e%2e%2f`, not to
    `../` -- so it must fail closed (file not found) rather than either
    silently succeeding or actually escaping the bundle directory."""
    src = tmp_path / "src"
    manifest = json.dumps({"files": {"%252e%252e%252f": {"sha256": "0" * 64, "bytes": 1}}})
    url = _publish_raw(src, manifest)
    dest = tmp_path / "dest"
    with pytest.raises(SystemExit):
        fetch(url, dest, verify=False)
    assert not any(dest.glob("*"))


@pytest.mark.parametrize(
    "manifest_text",
    [
        "[]",
        "{}",
        json.dumps({"files": []}),
        json.dumps({"files": {"a.txt": "oops"}}),
        json.dumps({"files": {"a.txt": {"bytes": 1}}}),
        json.dumps({"files": {"a.txt": {"sha256": 123}}}),
    ],
)
def test_malformed_manifest_shapes_raise_a_clear_systemexit(tmp_path, manifest_text):
    """A truncated or corrupted manifest download is plausible without any
    malice -- this module reports clearly everywhere else, so a bad shape
    here must not surface as a bare KeyError/AttributeError traceback."""
    src = tmp_path / "src"
    url = _publish_raw(src, manifest_text)
    dest = tmp_path / "dest"
    with pytest.raises(SystemExit, match="malformed"):
        fetch(url, dest)


def test_url_containment_check_is_independent_of_the_name_check():
    """`_resolve_fetch_url` alone -- not just `fetch`'s combination of checks
    -- must reject a traversal, so the URL-side guard is a real second line
    of defence and not merely code that the name check always beats it to."""
    base = "file:///tmp/bundle/src/manifest.json"
    with pytest.raises(SystemExit, match="outside its own directory"):
        _resolve_fetch_url(base, "%2e%2e%2fSECRET.txt")
    # a same-directory name is unaffected
    assert _resolve_fetch_url(base, "a.jsonl") == "file:///tmp/bundle/src/a.jsonl"


def test_a_nul_byte_in_a_name_is_rejected_not_a_bare_valueerror(tmp_path):
    src = tmp_path / "src"
    manifest = json.dumps({"files": {"a\x00.txt": {"sha256": "0" * 64, "bytes": 1}}})
    url = _publish_raw(src, manifest)
    dest = tmp_path / "dest"
    with pytest.raises(SystemExit, match="unsafe path"):
        fetch(url, dest)


# --- what the manifest's own `bytes` field is for ---------------------------


def test_a_manifest_entry_without_bytes_is_refused(tmp_path):
    """`bytes` is the only bound on how much a remote host can make this
    process write before the digest can say anything, so an entry without one
    is malformed rather than merely unusual."""
    src = tmp_path / "src"
    url = _publish_raw(
        src, json.dumps({"files": {"a.txt": {"sha256": "0" * 64}}})
    )
    with pytest.raises(SystemExit, match="malformed"):
        fetch(url, tmp_path / "dest")


@pytest.mark.parametrize("size", [True, -1, "8", 1.0, None])
def test_a_non_integer_or_negative_bytes_is_refused(tmp_path, size):
    """`True` is in this list on purpose: bool is an int subclass, so an
    unguarded isinstance check would read it as a one-byte cap."""
    src = tmp_path / "src"
    url = _publish_raw(
        src,
        json.dumps({"files": {"a.txt": {"sha256": "0" * 64, "bytes": size}}}),
    )
    with pytest.raises(SystemExit, match="malformed"):
        fetch(url, tmp_path / "dest")


def test_a_file_longer_than_its_declared_size_is_refused(tmp_path):
    src = tmp_path / "src"
    url = _publish(src, {"a.jsonl": b"eight..."})
    (src / "a.jsonl").write_bytes(b"eight..." + b"x" * 4096)
    dest = tmp_path / "dest"
    with pytest.raises(SystemExit, match="longer than"):
        fetch(url, dest, verify=False)
    assert not any(dest.glob("*"))


def test_a_file_shorter_than_its_declared_size_is_refused(tmp_path):
    """A truncated download that happens to be requested with --no-verify
    must still fail: the manifest said how long the file is."""
    src = tmp_path / "src"
    url = _publish(src, {"a.jsonl": b"the whole payload"})
    (src / "a.jsonl").write_bytes(b"the whole")
    dest = tmp_path / "dest"
    with pytest.raises(SystemExit, match="bytes but the manifest declares"):
        fetch(url, dest, verify=False)
    assert not any(dest.glob("*"))


# --- a bundle is all-or-nothing, not each file individually -----------------


def test_a_bundle_that_fails_late_leaves_no_earlier_file_behind(tmp_path):
    """The per-file guarantee does not compose into a per-bundle one.

    Before staging, a bundle failing at file 2 left file 1 sitting in `dest`
    with nothing recording that the set was incomplete -- and `locus doctor`
    reports a bundle as "present" from exactly that kind of evidence.
    """
    src = tmp_path / "src"
    url = _publish(src, {"a.jsonl": b"first file", "b.jsonl": b"second file"})
    (src / "b.jsonl").write_bytes(b"corrupted!!")  # same length, wrong bytes
    dest = tmp_path / "dest"
    with pytest.raises(SystemExit, match="sha256"):
        fetch(url, dest)
    assert not any(dest.glob("*")), (
        "a bundle that failed at its second file left its first behind"
    )


def test_a_successful_bundle_leaves_no_staging_directory(tmp_path):
    url = _publish(src := tmp_path / "src", {"a.jsonl": b"one", "b.jsonl": b"two"})
    assert src.exists()
    dest = tmp_path / "dest"
    fetch(url, dest)
    assert sorted(p.name for p in dest.iterdir()) == ["a.jsonl", "b.jsonl"]


# --- redirects: outside the confinement checks by construction --------------
#
# These are the only tests here that need a socket. `file://` has no notion of
# a redirect, so the one code path that cannot be exercised over it is exactly
# the one that slips past a check performed on the URL before it is opened.


class _Server:
    """A one-off localhost server whose routes the test writes.

    `routes` maps a path to either `("redirect", location)` or
    `("body", bytes)`. `written` counts the bytes actually handed to the
    socket, which is how the size-cap test tells "refused mid-stream" from
    "refused after the whole body arrived".
    """

    def __init__(self, routes):
        self.routes = routes
        self.written = 0
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args):  # keep pytest output clean
                pass

            def do_GET(self):
                kind, payload = outer.routes[self.path]
                if kind == "redirect":
                    self.send_response(302)
                    self.send_header("Location", payload)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    for i in range(0, len(payload), 65536):
                        block = payload[i : i + 65536]
                        self.wfile.write(block)
                        outer.written += len(block)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the client refused the rest, which is the point

        class Server(http.server.ThreadingHTTPServer):
            # http.server.HTTPServer.server_bind calls socket.getfqdn(), a
            # reverse DNS lookup that costs ~35 s on a laptop with no
            # resolver for 127.0.0.1. The value is only ever used to fill in
            # a Server: header, so it is set directly instead.
            def server_bind(self):
                socketserver.TCPServer.server_bind(self)
                self.server_name = "127.0.0.1"
                self.server_port = self.server_address[1]

        self._httpd = Server(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def close(self):
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture
def server():
    made = []

    def _make(routes):
        s = _Server(routes)
        made.append(s)
        return s

    yield _make
    for s in made:
        s.close()


def _manifest_body(entries):
    return json.dumps({"files": entries}).encode()


def test_a_redirect_is_refused_when_nothing_would_verify_the_bytes(
    server, tmp_path
):
    """`--no-verify` plus a redirect is an unbounded remote read.

    The confinement check runs on the URL this module resolves; a 302 replaces
    that URL afterwards, so it is outside the check by construction. With no
    sha256 to fall back on there is nothing left, so the redirect is refused.
    """
    payload = b"payload from somewhere else entirely"
    entry = {"a.txt": {"sha256": hashlib.sha256(payload).hexdigest(),
                       "bytes": len(payload)}}
    s = server({
        "/bundle/manifest.json": ("body", _manifest_body(entry)),
        "/bundle/a.txt": ("redirect", "/elsewhere/a.txt"),
        "/elsewhere/a.txt": ("body", payload),
    })
    dest = tmp_path / "dest"
    with pytest.raises(SystemExit, match="redirect"):
        fetch(f"{s.base}/bundle/manifest.json", dest, verify=False)
    assert not any(dest.glob("*"))


def test_a_redirect_is_followed_when_the_digest_still_has_to_match(
    server, tmp_path
):
    """Refusing redirects outright would be wrong: real deposit hosts answer
    file requests with a redirect to object storage. With hashing on, a
    redirect can deliver the manifest's bytes or nothing."""
    payload = b"payload from object storage"
    entry = {"a.txt": {"sha256": hashlib.sha256(payload).hexdigest(),
                       "bytes": len(payload)}}
    s = server({
        "/bundle/manifest.json": ("body", _manifest_body(entry)),
        "/bundle/a.txt": ("redirect", "/elsewhere/a.txt"),
        "/elsewhere/a.txt": ("body", payload),
    })
    dest = tmp_path / "dest"
    fetch(f"{s.base}/bundle/manifest.json", dest)
    assert (dest / "a.txt").read_bytes() == payload


def test_a_redirect_to_the_wrong_bytes_still_fails_the_digest(server, tmp_path):
    """The property that makes following a redirect acceptable at all."""
    entry = {"a.txt": {"sha256": hashlib.sha256(b"expected").hexdigest(),
                       "bytes": len(b"expected")}}
    s = server({
        "/bundle/manifest.json": ("body", _manifest_body(entry)),
        "/bundle/a.txt": ("redirect", "/elsewhere/a.txt"),
        "/elsewhere/a.txt": ("body", b"attacker"),  # same length
    })
    dest = tmp_path / "dest"
    with pytest.raises(SystemExit, match="sha256"):
        fetch(f"{s.base}/bundle/manifest.json", dest)
    assert not any(dest.glob("*"))


def test_a_redirect_loop_is_bounded_and_does_not_hang(server, tmp_path):
    entry = {"a.txt": {"sha256": "0" * 64, "bytes": 1}}
    s = server({
        "/bundle/manifest.json": ("body", _manifest_body(entry)),
        "/bundle/a.txt": ("redirect", "/bundle/a.txt"),
    })
    dest = tmp_path / "dest"
    with pytest.raises(SystemExit):
        fetch(f"{s.base}/bundle/manifest.json", dest)
    assert not any(dest.glob("*"))
    assert fetch_mod.MAX_REDIRECTS <= 10


def test_an_oversized_body_is_cut_off_mid_stream_not_after_it_lands(
    server, tmp_path
):
    """The cap has to bound what gets written, not report on it afterwards.

    The manifest declares 1 KiB; the host serves 16 MiB. `server.written`
    counts what the socket actually accepted, so a fetch that read the body to
    the end before checking would show all 16 MiB here.
    """
    body = b"x" * (16 << 20)
    entry = {"a.txt": {"sha256": "0" * 64, "bytes": 1024}}
    s = server({
        "/bundle/manifest.json": ("body", _manifest_body(entry)),
        "/bundle/a.txt": ("body", body),
    })
    dest = tmp_path / "dest"
    with pytest.raises(SystemExit, match="longer than"):
        fetch(f"{s.base}/bundle/manifest.json", dest)
    assert not any(dest.glob("*"))
    assert s.written < len(body), (
        f"the whole {len(body)}-byte body was accepted before the "
        "declared-size cap was applied"
    )
