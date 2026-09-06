import json
import pathlib
from pathlib import Path

import pytest

from locus.data.streaming import stream_dict, stream_list


def _write(tmp_path, obj):
    p = tmp_path / "d.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


class TestStreamDict:
    def test_yields_every_pair(self, tmp_path):
        obj = {f"k{i}": {"v": i} for i in range(50)}
        assert dict(stream_dict(_write(tmp_path, obj))) == obj

    def test_survives_braces_and_commas_inside_strings(self, tmp_path):
        obj = {"a": {"raw": 'he said "}{," and left'}, "b": {"raw": "plain"}}
        assert dict(stream_dict(_write(tmp_path, obj))) == obj

    def test_spans_buffer_boundaries(self, tmp_path):
        obj = {f"k{i}": {"raw": "x" * 500} for i in range(200)}
        got = dict(stream_dict(_write(tmp_path, obj), bufsize=256))
        assert got == obj

    def test_empty_dict_yields_nothing(self, tmp_path):
        assert list(stream_dict(_write(tmp_path, {}))) == []

    def test_missing_opening_brace_raises_systemexit_naming_the_path(self, tmp_path):
        p = tmp_path / "no_open.json"
        p.write_text("[1, 2]")
        with pytest.raises(SystemExit, match=str(p)):
            list(stream_dict(p))


# --- real-data smoke tests -------------------------------------------------
# The tests above round-trip through json.dumps, so they only prove the parser
# handles Python's own normalised output. These run against byte-faithful
# excerpts of the actual dataset files, which carry \uXXXX escapes and (in the
# contexts fixture) LaTeX braces inside string values -- the case that breaks
# brace-counting parsers. See fixtures/README.md.

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_NAMES = ["contexts_sample.json", "papers_sample.json"]


@pytest.mark.parametrize("name", FIXTURE_NAMES)
class TestAgainstRealData:
    def test_matches_json_load(self, name):
        p = FIXTURES / name
        assert dict(stream_dict(p)) == json.loads(p.read_text(encoding="utf-8"))

    @pytest.mark.parametrize("bufsize", [4096, 512, 64, 17])
    def test_matches_json_load_across_buffer_sizes(self, name, bufsize):
        # 17 bytes is smaller than one record, so every refill path is exercised
        p = FIXTURES / name
        assert dict(stream_dict(p, bufsize=bufsize)) == json.loads(
            p.read_text(encoding="utf-8")
        )


def test_contexts_fixture_still_exercises_braces_in_strings():
    # Guards the fixture itself: if a regeneration ever produces a fixture with
    # no LaTeX braces, the real-data tests silently stop testing the hard case.
    d = json.loads((FIXTURES / "contexts_sample.json").read_text(encoding="utf-8"))
    n = sum(
        1
        for v in d.values()
        if any(c in v["raw"] or c in v["masked_text"] for c in "{}")
    )
    assert n >= 10


class TestStreamList:
    FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "split_sample.json"

    def test_matches_json_load_on_real_data(self):
        with open(self.FIXTURE, encoding="utf-8") as f:
            expected = json.load(f)
        assert list(stream_list(self.FIXTURE)) == expected

    def test_survives_a_buffer_smaller_than_one_record(self, tmp_path):
        # The refill path is the whole point of the parser; a buffer that
        # cannot hold a single record exercises it on every element.
        p = tmp_path / "a.json"
        p.write_text(json.dumps([{"context_id": f"c{i}", "positive_ids": [i]} for i in range(20)]))
        assert list(stream_list(p, bufsize=17)) == json.loads(p.read_text())

    def test_empty_array(self, tmp_path):
        p = tmp_path / "e.json"
        p.write_text("[]")
        assert list(stream_list(p)) == []

    def test_empty_file_yields_nothing(self, tmp_path):
        p = tmp_path / "z.json"
        p.write_text("")
        assert list(stream_list(p)) == []

    def test_whitespace_between_records_is_skipped(self, tmp_path):
        p = tmp_path / "w.json"
        p.write_text('[\n  {"a": 1} ,\n\t{"a": 2}\n]')
        assert list(stream_list(p)) == [{"a": 1}, {"a": 2}]

    def test_truncated_mid_record_raises(self, tmp_path):
        p = tmp_path / "trunc_mid.json"
        p.write_text('[{"a": 1}, {"b": 2')  # truncated mid-record, no closing ]
        with pytest.raises(SystemExit):
            list(stream_list(p))

    def test_missing_closing_bracket_raises(self, tmp_path):
        p = tmp_path / "missing_bracket.json"
        p.write_text('[{"a": 1}, {"b": 2}')  # well-formed records but no closing ]
        with pytest.raises(SystemExit):
            list(stream_list(p))

    def test_well_formed_array_does_not_raise(self, tmp_path):
        # Guard against over-triggering: a properly-formed array should not raise
        p = tmp_path / "good.json"
        p.write_text('[{"a": 1}, {"b": 2}]')
        assert list(stream_list(p)) == [{"a": 1}, {"b": 2}]

    def test_bare_numbers_are_not_truncated_at_a_buffer_boundary(self, tmp_path):
        # dec.raw_decode succeeding exactly at the end of the buffer is
        # itself valid JSON for a bare number, so a tiny bufsize used to
        # split "1234567890" into "12345" and "67890" without ever raising.
        p = tmp_path / "nums.json"
        p.write_text("[1234567890, 22]")
        assert list(stream_list(p, bufsize=6)) == [1234567890, 22]

    def test_a_run_of_bare_integers_survives_a_tiny_buffer(self, tmp_path):
        p = tmp_path / "ints.json"
        nums = list(range(1000, 1030))
        p.write_text(json.dumps(nums))
        for bufsize in (4, 5, 6, 7, 8, 17):
            assert list(stream_list(p, bufsize=bufsize)) == nums

    def test_missing_opening_bracket_raises_systemexit_naming_the_path(self, tmp_path):
        p = tmp_path / "no_open.json"
        p.write_text('{"a": 1}')
        with pytest.raises(SystemExit, match=str(p)):
            list(stream_list(p))
