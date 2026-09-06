import pytest

from locus.core.hardness import is_easy, surname, title_containment


class TestSurname:
    def test_plain_string(self):
        assert surname("Jane Q. Doe") == "doe"

    def test_dict_form(self):
        assert surname({"name": "Jane Q. Doe"}) == "doe"

    def test_empty(self):
        assert surname("") == ""


class TestTitleContainment:
    def test_whole_title_present_scores_one(self):
        assert title_containment("gaia parallax zero point", "gaia parallax zero point") == 1.0

    def test_disjoint_scores_zero(self):
        assert title_containment("lattice chromodynamics", "stellar parallax") == 0.0

    def test_stopwords_do_not_create_overlap(self):
        assert title_containment("the and of a", "the and of a") == 0.0

    def test_partial(self):
        # half of the title's content words are on the surface
        assert title_containment("gaia parallax", "gaia offset") == pytest.approx(0.5)

    def test_is_invariant_to_window_length(self):
        # The whole point: Jaccard divides by |window u title|, so padding the
        # window drives the score to zero even though the title is fully
        # present and the item is plainly easy. Containment divides by
        # |title| and is unmoved.
        title = "gaia parallax zero point"
        short = "gaia parallax zero point offsets"
        padded = short + " " + " ".join(f"filler{i}" for i in range(200))
        assert title_containment(short, title) == 1.0
        assert title_containment(padded, title) == 1.0


class TestIsEasy:
    def test_surname_in_window_is_easy(self):
        assert is_easy("as Doe showed, the effect", "unrelated title", ["Jane Doe"], 0.5)

    def test_whole_title_in_a_long_window_is_easy(self):
        # 289 frozen rank items had the target's entire title in the window
        # and were still labelled hard, because Jaccard against a ~36-token
        # window cannot reach tau=0.1 for a 6-token title.
        title = "gaia parallax zero point"
        window = title + " " + " ".join(f"filler{i}" for i in range(200))
        assert is_easy(window, title, [], 0.5)

    def test_one_title_word_in_a_long_window_is_hard(self):
        window = "gaia " + " ".join(f"filler{i}" for i in range(200))
        assert not is_easy(window, "gaia parallax zero point", [], 0.5)

    def test_neither_is_hard(self):
        assert not is_easy(
            "the effect persists at low temperature", "gaia parallax", ["Jane Doe"], 0.5
        )
