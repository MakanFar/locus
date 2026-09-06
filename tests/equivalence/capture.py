# tests/equivalence/capture.py
"""Generate golden_digest.json from the CURRENT code. Run this exactly once,
against code known to be correct, and never again as a way of making a failing
gate pass -- that would silently bless the drift it exists to catch."""
import sys
from pathlib import Path

from locus import config

from . import golden


def main() -> int:
    out = Path(__file__).with_name("golden_digest.json")
    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 1
    golden.write_golden(out, golden.digest(config.WORK_DIR))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
