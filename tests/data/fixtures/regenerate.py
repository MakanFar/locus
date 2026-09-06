"""Rebuild the streaming-parser fixtures from the full dataset.

Copies each record's key/value span verbatim so the fixture keeps the source
file's own escaping and spacing. Round-tripping through json.dumps would
normalise away the very details these fixtures exist to test.

Usage:  .venv/bin/python -m tests.data.fixtures.regenerate
"""
import json
from collections.abc import Iterator
from pathlib import Path

from locus import config

OUT = Path(__file__).resolve().parent
SCAN_BYTES = 12 << 20


def iter_spans(path: Path, limit: int) -> Iterator[tuple[str, dict]]:
    """Yield (verbatim source span, decoded value) for the first `limit` records."""
    dec = json.JSONDecoder()
    with open(path, encoding="utf-8") as f:
        buf = f.read(SCAN_BYTES)
    i = buf.index("{") + 1
    for _ in range(limit):
        while buf[i] in " \n\t\r,":
            i += 1
        if buf[i] == "}":
            return
        try:
            _, j = dec.raw_decode(buf, i)
            while buf[j] in " \n\t\r:":
                j += 1
            val, k = dec.raw_decode(buf, j)
        except (ValueError, IndexError):
            return
        yield buf[i:k], val
        i = k


def write(name: str, chosen: list[str]) -> None:
    text = "{" + ",".join(chosen) + "}"
    json.loads(text)  # must be valid before it lands on disk
    (OUT / name).write_text(text, encoding="utf-8")
    print(f"{name}: {len(text)} bytes, {len(chosen)} records")


def main() -> None:
    braced, plain = [], []
    for span, val in iter_spans(config.contexts(), 4000):
        target = braced if any(
            c in val["raw"] or c in val["masked_text"] for c in "{}"
        ) else plain
        target.append(span)
    if len(braced) < 12:
        raise SystemExit(f"only {len(braced)} brace-bearing records found; widen SCAN_BYTES")
    write("contexts_sample.json", braced[:12] + plain[:28])

    write("papers_sample.json", [span for span, _ in iter_spans(config.papers(), 12)])


if __name__ == "__main__":
    main()
