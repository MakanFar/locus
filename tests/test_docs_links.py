"""Relative Markdown links must resolve, and the README must stay a front door."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MD = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
LINK = re.compile(r"\[[^\]]*\]\(([^)#]+)(?:#[^)]*)?\)")


@pytest.mark.parametrize("doc", MD, ids=lambda p: p.name)
def test_every_relative_link_resolves(doc):
    broken = []
    for target in LINK.findall(doc.read_text()):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        if not (doc.parent / target).exists():
            broken.append(target)
    assert not broken, f"{doc.name} has broken links: {broken}"


def test_the_readme_is_a_front_door_not_a_manual():
    lines = (ROOT / "README.md").read_text().splitlines()
    assert len(lines) < 150, (
        f"README is {len(lines)} lines; the normative content belongs in docs/"
    )


def test_the_readme_points_at_every_guide():
    readme = (ROOT / "README.md").read_text()
    for guide in ("schema.md", "reproduce.md", "custom-corpus.md", "scoring.md"):
        assert guide in readme, f"README never links docs/{guide}"
