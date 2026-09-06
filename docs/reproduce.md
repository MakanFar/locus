# Reproducing the paper

Three tiers. Each states what it needs, how long it takes and **the exact
numbers you should see**, so a failed reproduction is distinguishable from
a slow one.

| Tier | Needs | Time | Produces |
|---|---|---|---|
| 0 | the release bundle (~144 MB) | minutes, laptop | floors, export integrity, and the headline result once embeddings are on disk (see below) |
| 1 | + the upstream corpus, and one GPU-hour *or* Tier 0's precomputed vectors | hours | all four bases on both probes, including BM25 |
| 2 | + corpus-wide vectors (~2 GB) | hours | sequential completion, anchor influence, anchor analysis |

## Tier 0 — no corpus, no model, no GPU

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/locus doctor
.venv/bin/python -m pytest tests -q
```

**Expect no failures.** Some tests skip when optional extras (`torch`,
`gensim`, ...) are not installed; the exact skip count depends on your
environment, not on the artefact, so it is not quoted here. `locus doctor`
reports which optional extras import in your environment — that is the
environment-independent version of the same signal.

Then fetch the bundle and check it end to end:

```bash
.venv/bin/locus fetch work/release
```

`fetch` downloads the core bundle from Zenodo
(<https://doi.org/10.5281/zenodo.22260046>, version 0.2.0, CC-BY-4.0) and
verifies every file's sha256 against `bundle_manifest.json` before anything
lands; `--manifest-url` points it somewhere else. The bundle is **flat**:
`locus_rank_test.jsonl`, `locus_swap_test.jsonl`, `locus_anchors_test.jsonl`,
`locus_manifest_test.json` and `cocitation_train.npz` (plus the `val` split
and `bundle_manifest.json` itself) land directly in `work/release/` — there
is no `export/`
subdirectory. It does **not** carry `embeddings_test.parquet`: precomputed
embeddings are derived from upstream abstracts and are deliberately kept out
of the core bundle so its DOI survives independently of that separate
deposit: <https://doi.org/10.5281/zenodo.22432297> holds
`embeddings_{test,val}.parquet` (SPECTER2) plus the SciNCL and HAtten
vectors for both splits.

`fetch` already verified every file's sha256 while downloading it, so the
floors and counts below are exactly what the bundle's own manifest declares
— no rebuild, and no frozen pickles (which are not part of the bundle),
required:

```bash
python3 -c "
import json
m = json.load(open('work/release/locus_manifest_test.json'))
print(m['floors']); print(m['counts'])
"
```

**Expect exactly:**

| Quantity | Value |
|---|---|
| Rank chance floor | `0.2928968253968254` |
| Swap floor | `0.5`, with a zero-width bootstrap interval |
| Rank pools (test) | 55,488 |
| Headline pools | 35,701 |
| Contexts | 104,401 |
| Citing papers | 9,208 |

These are the manifest's declared floors, not a recomputation: `fetch`
already checked every file's sha256 against that manifest, and reading them
back here just confirms the print above matches what was verified on disk.
The floors are actually *recomputed* from pool construction, and checked by
**equality** (not tolerance), inside `locus build pipeline` and inside
`locus verify`'s null-control recomputation -- both of which need the
frozen pickles and are therefore Tier 1+, not Tier 0. See
[schema.md](schema.md) for what that equality check covers.

Then the headline itself. This is the one step Tier 0 cannot finish alone:
`--embeddings` names a vectors file that is not in the core bundle. Either
compute it yourself (Tier 1, one GPU-hour) or download it from the
precomputed-embeddings record:

```bash
curl -L -o work/embeddings_test.parquet \
  https://zenodo.org/records/22432297/files/embeddings_test.parquet
```

then:

```bash
.venv/bin/locus verify --export work/release \
    --embeddings work/embeddings_test.parquet \
    --graph      work/release/cocitation_train.npz
```

**Expect:** MRR `0.47771` → `0.59585` (`+0.11814`, [+0.11062, +0.12601]) and
R@1 `0.28051` → `0.43124` (`+0.15072`).

`verify` reads `locus_rank_test.jsonl` and `locus_anchors_test.jsonl` and
never opens the index pickle or any corpus text. That is enforced, not
described.

## Tier 1 — with the upstream corpus

Get the arXiv split of Local Citation Recommendation from Gu et al. (ECIR
2022); it is theirs and we do not redistribute it. Point `LOCUS_DATA_DIR` at
it — the layout is in [schema.md](schema.md).

```bash
export LOCUS_DATA_DIR=/path/to/lcr_arxiv LOCUS_WORK_DIR=$PWD/work
.venv/bin/locus build pipeline --split test
.venv/bin/locus build pipeline --split val     # exits 1; see below
.venv/bin/locus build graph
.venv/bin/locus build embed-inputs --split test
.venv/bin/locus embed --input work/embed_inputs_test.parquet \
                      --output work/embeddings_test.parquet
.venv/bin/locus sweep     --embeddings work/embeddings_val.parquet
.venv/bin/locus eval rank --embeddings work/embeddings_test.parquet
```

**`--split val` exits 1 and that is expected.** It fails one gate —
alignment failure 17.4% against a 15% threshold, versus 12.5% on test. Every
artefact is still written; the gate verdict comes after every `pickle.dump`.
The failures never reach a pool (all 19,585 are dropped by
`build_rank_items`), val and test share zero citing papers and zero
contexts, and the surviving val headline set is shaped like test's — 31,402
items across 2,074 papers, hard fraction 78.36% against test's 78.16%. The
gate was left as it is because it reports something true.

BM25 lives at this tier, not Tier 0, because it reads
`embed_inputs_{split}.parquet`, which carries upstream titles, abstracts and
context windows and is therefore not redistributable:

```bash
.venv/bin/locus sweep     --texts work/embed_inputs_val.parquet  --out work/sweep_val_bm25.json
.venv/bin/locus eval rank --texts work/embed_inputs_test.parquet --sweep work/sweep_val_bm25.json
```

**Expect** base MRR `0.50174` → `0.60399` (`+0.10226`).

**Expected wall clock:** the two `build pipeline` runs are ~5 minutes each
and need ~8 GB of RAM; `locus embed` is roughly one GPU-hour for both
splits, or several hours on CPU.

## Tier 2 — the remaining experiments

Needs corpus-wide vectors: 1,551,045 rows across seven parquet shards,
deposited separately from the core bundle so the core keeps its DOI if they
are ever withdrawn: <https://doi.org/10.5281/zenodo.22434185> (about 2.2 GB;
download the seven `embeddings_test_shard00*.parquet` files into
`work/embeddings_expb_test/`).

```bash
.venv/bin/locus exp completion --embeddings work/embeddings_expb_test
.venv/bin/locus exp influence  --embeddings work/embeddings_test.parquet
.venv/bin/locus exp anchors
```

**Expect** 22,274 locations in the completion experiment and 143,494
interventions in the influence pass.

`exp completion` prints two tables. The first is Recall@K counting the
revealed seed, which is what the published completion numbers are. The
second scores the same runs over `G(l)` minus that seed -- the members the
system still has to find -- because the first hands every arm `1/|G(l)|` for
free and 56.6% of locations have `|G(l)| = 2`. The base is z-scored over
the location's candidate list (`--standardize zscore`, the default and the
scale lambda was selected on; `--standardize none` reproduces the raw-cosine
numbers of earlier drafts). **Expect**, on the remaining members with a
revealed seed, base Recall@5 `0.13354` against `0.36023` rescored
(`+0.22669`), and `1.00000` reachability, since the oracle prefetch
guarantees the gold is a candidate.

Add `--task` to retrieve corpus-wide instead of reranking that prefetch:

```bash
.venv/bin/locus exp completion --embeddings work/embeddings_expb_test \
    --task work/expb_task_test_corpuswide.jsonl --out work/expb_corpuswide.json
```

Both commands load the seven shards into RAM, which is about 2.3 GB before
the key index. If that does not fit, build a memory-mapped store once and
pass its prefix instead -- the scorer picks the mmap path automatically when
`<prefix>_index.json` is present:

```bash
.venv/bin/python -m locus.scoring.mmapstore --shards work/embeddings_expb_test
```

That writes the prefix `work/embeddings_expb_test_mm`; pass it as
`--embeddings` and the resident set becomes the pages actually read (each
location touches only its own ~2,000 candidate rows) rather than the whole
matrix. The numbers are identical either way.

**Expect** `0.10043` against `0.25945` at Recall@5 and a reachability of
`0.69344` -- corpus-wide the gold is not guaranteed to be retrieved at all,
and that ceiling binds both arms equally.

## If a number does not match

Run `locus doctor` first — it reports Python version, which optional extras
import, what `LOCUS_DATA_DIR` and `LOCUS_WORK_DIR` point at, and whether a
bundle is present. It always exits 0, so it is safe to run when something
else has already failed.

See also [schema.md](schema.md) for the artefact layout, [scoring.md](scoring.md)
for the scorer protocol, and [custom-corpus.md](custom-corpus.md) for running
LOCUS against a different corpus.

## What the harness can run that the paper does not report

These exist and are tested, but no number in the paper comes from them:
`--agg masked_mean` / `--agg max` on `locus sweep` and `locus eval rank`
(alternative anchor-set aggregators, each with its own lambda sweep), the
node2vec null control (`python -m locus.scoring.node2vec --null-control`),
`locus build graph --splits train,val,test` (the deliberately leaky graph
behind the leakage row of the controls table), and
`locus exp completion --standardize none` (the raw-cosine base of earlier
drafts). Treat their outputs as ablations, not as reproductions.
