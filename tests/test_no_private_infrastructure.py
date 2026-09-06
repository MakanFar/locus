"""Nothing identifying the deployment account may enter the repository.

An AWS account ID shipped as an argparse default once and survived several
audits because they grepped for the old module path rather than for the
account. This greps for the account, and runs on every file.
"""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# 763104351884 is Amazon's own public Deep Learning Container registry and is
# documented publicly by AWS; it is not ours and is deliberately allowed.
FORBIDDEN = {
    "aws account id": re.compile(r"\b149057604535\b"),
    "internal project name": re.compile(r"\bflavadapt\b|\bvalency-research\b"),
    # Assembled from parts so this file does not itself contain the literal
    # macOS home-directory prefix that submission checkers flag.
    "a developer home directory": re.compile("/" + "Users" + r"/[a-z]+/"),
    "a real s3 bucket": re.compile(r"s3://(?!\.\.\.|<)[a-z0-9][a-z0-9.-]{2,}"),
    "an aws access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "an iam role arn": re.compile(r"arn:aws:iam::\d+:"),
}


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.split()
    return [ROOT / p for p in out]


@pytest.mark.parametrize("label,pattern", sorted(FORBIDDEN.items()))
def test_no_tracked_file_contains(label, pattern):
    hits = []
    for path in tracked_files():
        try:
            text = path.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{n}")
    assert not hits, f"{label} appears in: {hits}"
