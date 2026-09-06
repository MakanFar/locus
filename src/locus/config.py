"""Paths and frozen constants for the LOCUS harness. No logic lives here."""
import os
from pathlib import Path

# Must NOT be package-relative: `Path(__file__).parent` resolves inside the
# installed package (e.g. site-packages once this is pip-installed), so any
# driver run without LOCUS_WORK_DIR set would try to write frozen artefacts
# into the install itself. `Path.cwd()` is the conventional "relative to
# wherever you invoked the tool" default and can never do that.
WORK_DIR = Path(
    os.environ.get(
        "LOCUS_WORK_DIR", Path.cwd() / "work"
    )
)


def data_dir() -> Path:
    """The upstream corpus directory. Read-only; never written to."""
    raw = os.environ.get("LOCUS_DATA_DIR")
    if not raw:
        raise SystemExit(
            "LOCUS_DATA_DIR is not set. Set it to a directory holding the "
            "upstream corpus: contexts.json, papers.json, train.json, "
            "val.json, test.json. README.md, under 'Environment', says what "
            "each of those files has to contain."
        )
    return Path(raw)


def contexts() -> Path:
    return data_dir() / "contexts.json"


def papers() -> Path:
    return data_dir() / "papers.json"


def split_file(split: str) -> Path:
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split {split!r}")
    return data_dir() / f"{split}.json"


POOL_SIZE = 10       # G + 9 distractors. Ties are scored by their expectation
                     # under random tie-breaking, so there is one chance MRR
                     # floor and a constant scorer hits it exactly:
                     #   H_10/10 = 0.29289682539682540
SWAP_K = 5           # max swap pairs contributed by one citing paper
SHINGLE_N = 8        # word-shingle length for location clustering
THETA = 0.5          # Jaccard threshold for same-location
HARDNESS_TAU = 0.5   # fraction of the TARGET TITLE's content words that must
                     # appear in the window for a context to count as easy.
                     # Containment, not Jaccard: Jaccard divides by
                     # |window u title| and so tracks window length. At 0.5
                     # ('half the title is on the surface') the hard
                     # fraction is 77.4% on test, inside the 40-85% band.
SEED = 0

# --- Build gate thresholds, as published in README.md's *Gates* table ------
#
# These lived as bare literals inside `build/pipeline.py`, where editing one
# moved no test at all -- the same defect the equivalence gate had for the six
# constants above. They are here so `tests/equivalence/test_frozen_config.py`
# can pin them against the published table.
#
# The first two state what the benchmark IS and hold for any corpus. The last
# two are statements about the size and composition of Gu et al.'s test split,
# which is why `build/pipeline.py --corpus` reports them instead of failing on
# them -- a corpus of 320 contexts cannot have a 30,000-item headline set and
# is not defective for it.
MAX_ALIGNMENT_FAILURE = 0.15   # expect 8-11%; 12.50% on test, 17.37% on val
MIN_ANCHOR_COVERAGE = 0.60     # |A|>=1 over the union of both anchor routes
HARD_FRACTION_BAND = (0.40, 0.85)   # 78.16% on test
MIN_HEADLINE = 30_000               # 35,701 on test
