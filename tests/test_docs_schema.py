"""docs/schema.md is normative, so it is pinned to the code it describes.

The field tables are parsed out of the Markdown and compared against real
module-level constants. A field renamed in `records.py` and not in the doc
fails here, which is the only thing that keeps a normative document honest
over time. This is the same discipline as
tests/equivalence/test_frozen_config.py: read the source of truth, never
restate it.
"""
import re
from pathlib import Path

from locus.data import ingest, records

DOC = Path(__file__).resolve().parents[1] / "docs" / "schema.md"


def fields_in_table(doc: str, heading: str) -> list[str]:
    """First-column `code` spans of the first Markdown table after `heading`.

    Stops at the next heading so two adjacent tables cannot bleed together.
    """
    after = doc.split(heading, 1)[1]
    section = re.split(r"\n#{2,} ", after, maxsplit=1)[0]
    return re.findall(r"^\| *`([^`]+)` *\|", section, re.MULTILINE)


def test_context_row_fields_match_records_REQUIRED():
    doc = DOC.read_text()
    # The contract is written as `masked_text` on disk and renamed to
    # `masked` by jsonl_source, so the doc names the on-disk field.
    on_disk = tuple(f if f != "masked" else "masked_text" for f in records.REQUIRED)
    assert tuple(fields_in_table(doc, "### `contexts.jsonl`")) == on_disk


def test_paper_row_fields_match_records_PAPER_FIELDS():
    doc = DOC.read_text()
    assert tuple(fields_in_table(doc, "### `papers.jsonl`")) == (
        "id", *records.PAPER_FIELDS
    )


def test_the_corpus_filenames_are_the_ones_ingest_looks_for():
    doc = DOC.read_text()
    for name in (ingest.CONTEXTS_FILE, ingest.PAPERS_FILE, ingest.SPLITS_DIR):
        assert name in doc, f"schema.md never mentions {name!r}"
    for split in ingest.SPLITS:
        assert f"{split}.json" in doc


def test_the_documented_export_schema_version_is_the_one_written():
    from locus.data import export

    doc = DOC.read_text()
    assert f'"schema": {export.SCHEMA}' in doc
