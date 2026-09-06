import pathlib
import subprocess
import sys

from locus.core.locations import cluster, shingles

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


class TestShingles:
    def test_digits_and_case_are_normalised_away(self):
        a = shingles("The Effect Of [4] On Gaia DR2 Parallaxes Holds Here Now", n=3)
        b = shingles("the effect of [17] on gaia dr2 parallaxes holds here now", n=3)
        assert a == b

    def test_short_text_yields_empty_set(self):
        assert shingles("only three words", n=8) == frozenset()

    def test_stable_across_interpreter_hash_seeds(self):
        # The index is the root of every frozen artefact and the plan commits
        # to "every artefact-producing function is deterministic". Python's
        # builtin hash() of a str is PYTHONHASHSEED-randomised, so keying
        # shingles on it makes collisions -- and therefore Jaccard, cluster
        # labels, gold sets and pools -- vary between processes.
        text = "we adopt the gaia dr2 parallax zero point offset reported in that work"
        code = (
            f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r});"
            "from locus.core.locations import shingles;"
            f"print(sorted(shingles({text!r})))"
        )
        runs = {
            subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, check=True,
                env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            ).stdout
            for seed in ("0", "1", "12345")
        }
        assert len(runs) == 1


class TestCluster:
    def test_identical_texts_share_a_label(self):
        t = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        assert cluster([t, t]) == [0, 0]

    def test_disjoint_texts_get_separate_labels(self):
        a = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        b = "one two three four five six seven eight nine ten eleven"
        assert cluster([a, b]) == [0, 1]

    def test_same_passage_with_moved_marker_clusters(self):
        a = "we adopt the parallaxes) [4] [5] and gaia dr2 zero point offset from"
        b = "we adopt the parallaxes) [4] [5] and gaia dr2 zero point offset from"
        c = "an entirely unrelated sentence about lattice quantum chromodynamics here"
        assert cluster([a, b, c]) == [0, 0, 1]

    def test_labels_are_first_appearance_ordered(self):
        a = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        b = "one two three four five six seven eight nine ten eleven"
        assert cluster([b, a, b]) == [0, 1, 0]

    def test_empty_input(self):
        assert cluster([]) == []
