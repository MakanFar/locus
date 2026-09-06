# LOCUS

A within-paper citation-location discrimination benchmark: given a citation
context, which paper belongs *here* rather than elsewhere in the same citing
paper.

## Install

    python3.11 -m venv .venv
    .venv/bin/pip install -e ".[dev,node2vec]"

That installs the `locus` command into `.venv/bin/`; every command below is
written as `.venv/bin/locus`, so it works without activating anything.

Python 3.11+; `[node2vec]` adds gensim for one ablation and `[cloud]` adds
boto3 for `locus.embed_specter2` (see [docs/scoring.md](docs/scoring.md)).

Run `.venv/bin/locus doctor` first. It reports Python version, which
optional extras (`torch`, `transformers`, `gensim`, `boto3`) are importable,
whether `LOCUS_DATA_DIR`/`LOCUS_WORK_DIR` are set and what they point at, and
whether an artefact bundle is present -- and always exits 0, on a bare clone
with nothing configured included, so it is the thing to run when `locus
build` or `locus verify` fails and it isn't obvious why.

## CLI

    locus doctor                 report the environment (see Install above)
    locus fetch <dest>           download the released artefact bundle from
                                 Zenodo and check
                                 every file against its manifest.
    locus ingest <root>          validate a JSONL corpus directory
                                 (see docs/custom-corpus.md)
    locus build {pipeline,graph,embed-inputs,export}
                                 `pipeline`, `graph` and `embed-inputs` take
                                 `--corpus DIR` to read a JSONL corpus instead
                                 of `$LOCUS_DATA_DIR`
    locus bundle --work DIR --out DIR
                                 assemble an export plus the co-citation graph
                                 into the flat, ID-only deposit `locus fetch`
                                 downloads, with a verifying manifest
    locus embed                  a local, credential-free embedder
    locus eval {rank,swap,all}
                                 `--embeddings` scores with a dense encoder;
                                 `--texts` scores the same pools with BM25
    locus sweep                  sweep lambda on val
    locus verify                 recompute the headline from an export
    locus exp {completion,influence,anchors}



## Layout

    src/locus/core/          what the benchmark is: pools, floors, aggregation
    src/locus/data/          corpora in, release files out
      ingest.py              the JSONL corpus contract, and its validator
      adapters/gu2022.py     upstream corpus -> that contract
      corpus.py              contexts and papers -> the frozen index
      export.py              frozen artefacts -> the portable, ID-only release
      fetch.py               download a released bundle, verified end to end
    src/locus/scoring/       anything that scores a (context, candidate) pair
    src/locus/build/         corpus -> frozen artefacts, and the gates
    src/locus/eval/          artefacts + scorer -> reported numbers
    src/locus/experiments/   paper reproduction, not benchmark surface

Dependencies run downward only; `tests/test_layering.py` enforces it.

## Environment

    LOCUS_DATA_DIR   upstream corpus (read-only)
    LOCUS_WORK_DIR   frozen artefacts and reports; defaults to ./work

`LOCUS_DATA_DIR` has no default: anything that reads the corpus fails with a
message naming the variable rather than guessing a path. The corpus layout it
must point at is in [docs/schema.md](docs/schema.md).

## Documentation

| Guide | For |
|---|---|
| [docs/reproduce.md](docs/reproduce.md) | reproducing the paper's results, in three tiers |
| [docs/schema.md](docs/schema.md) | the normative data contracts, in and out |
| [docs/custom-corpus.md](docs/custom-corpus.md) | building LOCUS from your own corpus |
| [docs/scoring.md](docs/scoring.md) | scoring LOCUS with your own model |

## Release status

See [CHANGELOG.md](CHANGELOG.md) for what shipped in each release.

## Tests

```bash
.venv/bin/python -m pytest tests -q          # the suite; no data needed
.venv/bin/ruff check src tests
```

Tests that recompute frozen numbers are marked `equivalence` and skip unless
`LOCUS_WORK_DIR` holds the artefacts:

```bash
LOCUS_WORK_DIR=/path/to/work .venv/bin/python -m pytest tests -m equivalence
```
