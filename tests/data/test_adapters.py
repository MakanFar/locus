"""The adapter must produce a corpus its own validator accepts."""
import json

from locus.data.adapters.gu2022 import convert
from locus.data.ingest import validate


def test_convert_produces_a_valid_corpus(tmp_path):
    src = tmp_path / "upstream"
    src.mkdir()
    (src / "contexts.json").write_text(json.dumps({
        "c1": {"raw": "we build on [1] and [2]", "masked_text": "we build on  TARGETCIT  OTHERCIT",
               "citing_id": "p1", "refid": "r1"},
        "c2": {"raw": "we build on [1] and [2]", "masked_text": "we build on OTHERCIT  TARGETCIT ",
               "citing_id": "p1", "refid": "r2"},
    }))
    (src / "papers.json").write_text(json.dumps({
        "r1": {"title": "A", "abstract": "aa"}, "r2": {"title": "B", "abstract": "bb"},
    }))
    for name, ids in (("train", []), ("val", []), ("test", ["c1", "c2"])):
        (src / f"{name}.json").write_text(json.dumps(ids))

    out = tmp_path / "converted"
    manifest = convert(src, out)
    assert manifest["contexts"] == 2
    assert validate(out) == []
