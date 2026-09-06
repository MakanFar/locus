"""Incremental parser for the dataset's multi-GB single-line JSON dicts.

`contexts.json` and `papers.json` are one enormous line each, so `json.load`
would need the whole file resident. This walks the file with a sliding buffer
and `raw_decode`, yielding one (key, value) pair at a time.
"""
import json
from collections.abc import Iterator
from pathlib import Path

_SKIP = " \n\t\r,"


def stream_dict(path: str | Path, bufsize: int = 1 << 22) -> Iterator[tuple[str, dict]]:
    dec = json.JSONDecoder()
    with open(path, encoding="utf-8") as f:
        buf = f.read(bufsize)
        if not buf:
            return
        try:
            i = buf.index("{") + 1
        except ValueError:
            raise SystemExit(
                f"{path} does not start with a JSON object: no '{{' found in "
                f"the first {len(buf)} bytes read"
            ) from None
        while True:
            while True:
                while i < len(buf) and buf[i] in _SKIP:
                    i += 1
                if i < len(buf) and buf[i] == "}":
                    return
                try:
                    key, j = dec.raw_decode(buf, i)
                    while buf[j] in " \n\t\r:":
                        j += 1
                    # A value decode that lands exactly at the buffer's end
                    # has the same "is it really complete, or just where the
                    # buffer happened to stop" ambiguity that stream_list
                    # guards against below -- but every value here is a
                    # dict/context record, and callers only ever decode
                    # objects, which raw_decode cannot mistake for complete
                    # until the closing '}' is actually seen.
                    val, k = dec.raw_decode(buf, j)
                    break
                except (ValueError, IndexError):
                    chunk = f.read(bufsize)
                    if not chunk:
                        return
                    buf = buf[i:] + chunk
                    i = 0
            yield key, val
            i = k
            if i > bufsize:
                buf = buf[i:]
                i = 0


def stream_list(path: str | Path, bufsize: int = 1 << 22) -> Iterator[dict]:
    """Same sliding-buffer walk as stream_dict, for a top-level JSON array.

    The split files (train/val/test.json) are arrays of ~3M small records;
    json.load on train.json builds every one of them before the caller sees
    the first.
    """
    dec = json.JSONDecoder()
    with open(path, encoding="utf-8") as f:
        buf = f.read(bufsize)
        if not buf:
            return
        try:
            i = buf.index("[") + 1
        except ValueError:
            raise SystemExit(
                f"{path} does not start with a JSON array: no '[' found in "
                f"the first {len(buf)} bytes read"
            ) from None
        while True:
            while True:
                while i < len(buf) and buf[i] in _SKIP:
                    i += 1
                if i < len(buf) and buf[i] == "]":
                    return
                try:
                    val, k = dec.raw_decode(buf, i)
                except (ValueError, IndexError):
                    chunk = f.read(bufsize)
                    if not chunk:
                        raise SystemExit(
                            f"{path} is truncated or malformed; the array was opened "
                            "but EOF reached without a closing `]`"
                        ) from None
                    buf = buf[i:] + chunk
                    i = 0
                    continue
                if k == len(buf):
                    # raw_decode succeeding exactly at the end of the buffer
                    # is itself valid JSON for a bare number (unlike a dict
                    # or string, which need a closing delimiter we would not
                    # yet have seen) -- so this could be a truncation, not a
                    # complete value. Refill from the same start position and
                    # retry before trusting it; only accept it once refilling
                    # proves there is genuinely no more data to extend it.
                    chunk = f.read(bufsize)
                    if chunk:
                        buf = buf[i:] + chunk
                        i = 0
                        continue
                break
            yield val
            i = k
            if i > bufsize:
                buf = buf[i:]
                i = 0
