# Test fixtures

Byte-faithful excerpts of the Gu et al. (ECIR 2022) arXiv local-citation-recommendation
dataset, used so the streaming parser is tested against the file's *actual* serialization
rather than against Python's `json.dumps` output.

| File | Records | Contents |
|---|---:|---|
| `contexts_sample.json` | 40 | 12 records whose `raw`/`masked_text` contain `{` or `}` (LaTeX in the citation window — the case that breaks brace-counting parsers), plus 28 ordinary ones |
| `papers_sample.json` | 12 | First 12 paper records: nested `authors` list, mixed types |
| `split_sample.json` | 40 | First 40 split-file records: JSON array, each with `context_id` and `positive_ids` |

Each record's key/value span is copied verbatim from the source file, preserving its original
escaping (`\uXXXX` sequences appear in both) and spacing. Only the joining commas and the outer
braces are synthesised.

### split_sample.json

First 40 entries of the real `test.json`, verbatim. A JSON **array** (the split
files are arrays; `contexts.json` and `papers.json` are objects), so it is the
fixture for `stream_list`. Each entry has `context_id` and `positive_ids`.

## Regenerating

Requires the full upstream dataset, with `LOCUS_DATA_DIR` pointing at it
(e.g. `/path/to/local-citation-recommendation/arxiv`). There is no default:
`config.data_dir()` raises when the variable is unset.

```bash
.venv/bin/python -m tests.data.fixtures.regenerate
```

Do not regenerate by round-tripping through `json.dumps` — that would normalise exactly the
serialisation details these fixtures exist to test.
