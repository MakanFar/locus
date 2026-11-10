"""Check that a number printed in the paper is the number a report holds.

`reports/paper_map.json` lists claims. Each names a report file, a dotted key
into it, the value exactly as the paper prints it, and -- required -- the
fields that say what the value is a measurement of: how many items, how many
citing papers, which averaging unit, which anchor route, which threshold.

The last part is the point. A report recomputed on a different subset can be
swapped for the right one and still be internally consistent, and a check
that compares digits alone passes it. Here a claim fails when the digits
match and the definition does not.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_SEPARATORS = re.compile(r"\{,\}|,|\\,")
# U+2212 MINUS SIGN, which a typeset paper uses where a report has a hyphen.
_MINUS = chr(0x2212)


def lookup(report, key: str):
    """Value at a dotted path; integer segments index into lists.

    A dict key may itself contain a dot (a threshold such as "0.3"), so at a
    dict the longest run of segments that is a key is taken.
    """
    node = report
    parts = key.split(".")
    i = 0
    while i < len(parts):
        if isinstance(node, list):
            node = node[int(parts[i])]
            i += 1
            continue
        for j in range(len(parts), i, -1):
            candidate = ".".join(parts[i:j])
            if candidate in node:
                node, i = node[candidate], j
                break
        else:
            raise KeyError(key)
    return node


def _number(printed: str) -> tuple[float, int]:
    """(value, decimals) of a number as the paper prints it."""
    text = _SEPARATORS.sub("", printed).replace(_MINUS, "-").strip()
    decimals = len(text.split(".")[1]) if "." in text else 0
    return float(text), decimals


def check_claim(claim: dict, reports: Path) -> list[str]:
    """Problems with one claim; empty when it holds."""
    name = claim.get("id", "?")
    path = Path(reports) / claim["report"]
    if not path.exists():
        return [f"{name}: report {claim['report']} is missing"]
    report = json.loads(path.read_text())

    problems: list[str] = []
    try:
        actual = lookup(report, claim["key"])
    except (KeyError, IndexError, TypeError, ValueError):
        problems.append(f"{name}: {claim['report']} has no key {claim['key']}")
    else:
        want, decimals = _number(claim["paper"])
        got = round(float(actual) * claim.get("scale", 1), decimals)
        if got != want:
            problems.append(
                f"{name}: paper prints {claim['paper']} but {claim['report']}:"
                f"{claim['key']} is {actual!r}")

    expect = claim.get("expect")
    if not expect:
        problems.append(
            f"{name}: no `expect` fields -- a claim must pin down what its "
            "number is measured on, not only its digits")
        return problems
    for key, want in expect.items():
        try:
            got = lookup(report, key)
        except (KeyError, IndexError, TypeError, ValueError):
            problems.append(f"{name}: {claim['report']} has no field {key}")
            continue
        if got != want:
            problems.append(
                f"{name}: {claim['report']}:{key} is {got!r}, the claim "
                f"expects {want!r}")
    return problems


def check_map(claims: dict, reports: Path, tex: str | None = None) -> list[str]:
    """Problems across every claim, and, given the paper source, any printed
    value that does not occur in it."""
    problems: list[str] = []
    for claim in claims["claims"]:
        problems.extend(check_claim(claim, reports))
        if tex is not None and claim["paper"] not in tex:
            problems.append(
                f"{claim.get('id', '?')}: {claim['paper']} not found in the paper")
    return problems


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", type=Path, required=True)
    ap.add_argument("--reports", type=Path, default=None,
                    help="directory of report files (default: the map's own)")
    ap.add_argument("--tex", type=Path, default=None,
                    help="paper source; also check each printed value occurs in it")
    args = ap.parse_args(argv)
    claims = json.loads(args.map.read_text())
    tex = args.tex.read_text() if args.tex else None
    problems = check_map(claims, args.reports or args.map.parent, tex)
    for line in problems:
        print(line)
    print(f"{len(claims['claims'])} claims, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
