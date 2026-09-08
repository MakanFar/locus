# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `locus fetch` defaults to the deposited core bundle
  (doi:10.5281/zenodo.22260046); `--manifest-url` is now an override.
- `reports/`: the run reports every number in the paper is read from, with a
  table/figure -> file map in `reports/README.md`.
- `locus exp anchors` writes the Figure 4 anchor map (`anchor_map_{tag}.json`
  and a pgfplots table) from the influence rows.
- `locus eval rank` reports the coverage stratification (Table 6) and the
  other-location control paired against the base; `locus.eval.controls`
  pairs every anchor arm against the base.
- `locus exp completion --standardize {zscore,none}`: the base is z-scored
  over the candidate list by default, which is Eq. 1 and the scale lambda was
  selected on. `none` reproduces the raw-cosine numbers of earlier drafts.
- `locus embed --revision`, with per-model pins (`MODEL_REVISIONS`) so
  `--model malteos/scincl` no longer loads at SPECTER2's commit.

### Changed
- `docs/reproduce.md` now gives a command and the expected numbers for every
  row in the paper: both probes on all four bases, the controls, the raw
  co-count, node2vec and sibling-titles ablations, the leaky graph, the
  corpus-wide retrieval step, and the memory constraint on running the
  embedder and the completion experiment together.
- `build/` is gitignored: a non-editable `pip install .` writes it into the
  source tree.
- Report files are named for what they hold: `build_report_{split}.json`
  (was `day1_report_`) and `rank_report*.json` (was `day3_report*`).
- Sequential-completion intervals use 1000 bootstrap resamples, like every other
  interval in the paper (was 500).

### Removed
- `CITATION.cff`: citation metadata is deferred until the paper has a
  citable form; the Zenodo records carry their own.
- `locus.scoring.adapt` (MNRL adapter) and `locus.experiments.intent` with
  the `locus exp intent` verb: neither backs a result in the paper, and the
  adapter depended on a training script that is not in this repository.

## [0.2.0] - 2026-08-31

### Added
- `locus` CLI over the existing entry points (18 verbs).
- `locus bundle` — assemble an export plus the co-citation graph into the
  flat, ID-only deposit `locus fetch` downloads, with a verifying manifest.
- A documented JSONL corpus contract (`locus.data.ingest`) with a
  per-invariant validator, and `locus.data.adapters.gu2022` converting the
  upstream corpus into it.
- `locus.scoring.bm25` — Okapi BM25 as a lexical base. Every eval entry
  point takes `--texts` in place of `--embeddings`.
- `locus.scoring.embed` — a local, credential-free SPECTER2 embedder.
- `locus.data.fetch` — release-bundle download verified against a manifest.
- `LICENSE`, `NOTICE`, `CITATION.cff` and `docs/`.

### Changed
- Extracted from a monorepo into a standalone package with history preserved.
- Layered subpackages (`core`, `data`, `scoring`, `build`, `eval`,
  `experiments`) with an import-direction test.
- The floor gate now tests its inputs rather than their aggregate, so
  it is exact for every group size rather than by luck of this corpus's.

### Fixed
- `requirements.txt` omitted `polars` and `scipy`; dependencies now live in
  `pyproject.toml` and a CI job installs from it into a clean environment.
- `config.py` hardcoded an absolute corpus path; it is now `LOCUS_DATA_DIR`
  with an error naming the variable.

## [0.1.0-alpha] - 2026-08-28

Initial extraction.
