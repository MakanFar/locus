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
.venv/bin/locus build embed-inputs --split val
.venv/bin/locus embed --input work/embed_inputs_test.parquet --output work/embeddings_test.parquet
.venv/bin/locus embed --input work/embed_inputs_val.parquet  --output work/embeddings_val.parquet
.venv/bin/locus sweep     --embeddings work/embeddings_val.parquet
.venv/bin/locus eval rank --embeddings work/embeddings_test.parquet
.venv/bin/locus eval swap --val-embeddings work/embeddings_val.parquet \
                          --test-embeddings work/embeddings_test.parquet --tag specter2
```

Both `embeddings_{val,test}.parquet` are also in the embeddings deposit
(<https://doi.org/10.5281/zenodo.22432297>) if you would rather not run
`locus embed`. **Expect** on Rank `0.47771` → `0.59585` (`+0.11814`) and on
Swap `0.81617` → `0.84369` (`+0.02753`, lambda 0.2); `eval rank` also prints
the coverage strata (Table 6) and the other-location control against the
base on every run.

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

**Expect** base MRR `0.50174` → `0.60399` (`+0.10226`), and on Swap
(`--val-texts`/`--test-texts` in place of the embeddings flags, `--tag bm25`)
`0.83399` → `0.85879` (`+0.02480`, lambda 0.1).

### The remaining rows of Tables 1, 2 and 7

SciNCL and HAtten need their own vectors: [scoring.md](scoring.md) has the
`locus embed --model malteos/scincl` and `locus.experiments.hatten` commands,
and the embeddings deposit carries both for both splits. Then the same three
commands per base, with `--out`/`--tag` so nothing overwrites the SPECTER2
reports:

```bash
for base in scincl hatten; do
  .venv/bin/locus sweep     --embeddings work/embeddings_${base}_val.parquet  --out work/sweep_val_$base.json
  .venv/bin/locus eval rank --embeddings work/embeddings_${base}_test.parquet --sweep work/sweep_val_$base.json --out work/rank_report_$base.json
  .venv/bin/locus eval swap --val-embeddings work/embeddings_${base}_val.parquet \
      --test-embeddings work/embeddings_${base}_test.parquet --sweep work/sweep_val_$base.json --tag $base
done
```

**Expect** Rank SciNCL `0.47019` → `0.59140` and HAtten `0.46643` → `0.57608`;
Swap SciNCL `0.80090` → `0.83377` and HAtten `0.83073` → `0.85290`.

The controls table, all on SPECTER2:

```bash
.venv/bin/locus eval rank --embeddings work/embeddings_test.parquet --slice easy --out work/rank_report_easy.json
.venv/bin/python -m locus.eval.controls --embeddings work/embeddings_test.parquet
.venv/bin/locus sweep     --embeddings work/embeddings_val.parquet  --assoc count --out work/sweep_val_count.json
.venv/bin/locus eval rank --embeddings work/embeddings_test.parquet --assoc count --sweep work/sweep_val_count.json --out work/rank_report_count.json
.venv/bin/locus build graph --splits train,val,test          # the deliberately leaky graph -> work/cocitation_full.npz
.venv/bin/locus sweep     --embeddings work/embeddings_val.parquet  --graph work/cocitation_full.npz --out work/sweep_val_fullgraph.json
.venv/bin/locus eval rank --embeddings work/embeddings_test.parquet --graph work/cocitation_full.npz --sweep work/sweep_val_fullgraph.json --out work/rank_report_fullgraph.json
```

**Expect** easy slice `0.65783` → `0.68738` (`+0.02956`); random anchors
`0.47741` and degree-matched `0.47240` against the `0.47771` base; raw
co-counts `+0.10272` at lambda 7; the leaky graph `0.80729` (`+0.32959`),
the 2.8x inflation the paper warns about.

The sibling-titles control appends the anchors' titles to the query, so it
needs its own embeddings, and no deposit carries them:

```bash
.venv/bin/locus build embed-inputs --split val  --variant siblings
.venv/bin/locus build embed-inputs --split test --variant siblings
.venv/bin/locus embed --input work/embed_inputs_siblings_val.parquet  --output work/embeddings_siblings_val.parquet
.venv/bin/locus embed --input work/embed_inputs_siblings_test.parquet --output work/embeddings_siblings_test.parquet
.venv/bin/locus sweep     --embeddings work/embeddings_siblings_val.parquet  --out work/sweep_val_siblings.json
.venv/bin/locus eval rank --embeddings work/embeddings_siblings_test.parquet --sweep work/sweep_val_siblings.json --out work/rank_report_siblings.json
```

**Expect** base `0.50864` → `0.60007` (`+0.09143`).

**Expected wall clock and memory:** the two `build pipeline` runs are under
a minute each on a fast disk (~5 minutes on a slow one) and need ~8 GB of
RAM; `build graph` about 5 minutes, the leaky one 7; every sweep, rank, swap
and control under a minute. `locus embed` is roughly one GPU-hour for both
splits, or several hours on CPU. It holds the model and the whole input
frame, so **run it on its own**: alongside the Tier 2 experiments on a 16 GB
laptop the two swap each other out and neither makes progress.

## Tier 2 — the remaining experiments

Needs corpus-wide vectors: 1,551,045 rows across seven parquet shards,
deposited separately from the core bundle so the core keeps its DOI if they
are ever withdrawn: <https://doi.org/10.5281/zenodo.22434185> (about 2.2 GB;
download the seven `embeddings_test_shard00*.parquet` files into
`work/embeddings_expb_test/`).

Build the memory-mapped store first: the completion runs touch only each
location's ~2,000 candidate rows, so the resident set stays small, whereas
loading the seven shards into RAM costs about 2.3 GB before the key index.
Then the candidate space, the two oracle-prefetch completion runs, the
leave-one-anchor-out pass and the anchor analysis, **one at a time**:

```bash
.venv/bin/python -m locus.scoring.mmapstore --shards work/embeddings_expb_test      # -> work/embeddings_expb_test_mm
.venv/bin/python -m locus.experiments.completion_inputs --split test                # candidate space from the oracle prefetch
.venv/bin/locus exp completion --embeddings work/embeddings_expb_test_mm --mode revealed --out work/expb_report_revealed.json
.venv/bin/locus exp completion --embeddings work/embeddings_expb_test_mm --mode selfseed --out work/expb_report_selfseed.json
.venv/bin/locus exp influence  --embeddings work/embeddings_test.parquet --sim-store work/embeddings_expb_test_mm --dump work/influence_rows_specter2.pkl
.venv/bin/locus exp anchors --tag specter2
```

**Expect** 22,274 locations per completion run (about 30 minutes each);
self-seeded Recall@20 `0.26383` → `0.40162` (`+0.13779`), which is Table 3
setting B; 143,494 interventions in the influence pass, categorised
12,666 informative / 44,272 redundant / 86,556 non-co-cited; and
`anchor_map_specter2.json` plus its pgfplots table for Figure 4.

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

Corpus-wide retrieval replaces the oracle prefetch with the top 2,000
papers by cosine over the whole store; the retrieval step writes the task
file the completion runs then take through `--task`:

```bash
.venv/bin/python -m locus.scoring.retrieve --store work/embeddings_expb_test_mm --k 2000   # -> expb_task_test_corpuswide.jsonl, retrieve_report_test.json
.venv/bin/locus exp completion --embeddings work/embeddings_expb_test_mm --mode selfseed \
    --task work/expb_task_test_corpuswide.jsonl --out work/expb_report_corpuswide.json
.venv/bin/locus exp completion --embeddings work/embeddings_expb_test_mm --mode revealed \
    --task work/expb_task_test_corpuswide.jsonl --out work/expb_report_revealed_corpuswide.json
```

**Expect** first-stage recall over 67,492 gold citations of `0.31333` at
k=100 and `0.68381` at k=2000; self-seeded Recall@20 `0.19377` → `0.28915`
(`+0.09538`), which is Table 3 setting C; and, on the remaining members with
a revealed seed, `0.10043` against `0.25945` at Recall@5 with a reachability
of `0.69344` -- corpus-wide the gold is not guaranteed to be retrieved at
all, and that ceiling binds both arms equally.

The node2vec rows replace PPMI with a learned graph embedding. The paper's
row uses `p = 1, q = 8`, selected by validation MRR over
`q in {0.5, 1, 2, 4, 8}`; each `(p, q)` is one training run and one sweep:

```bash
.venv/bin/python -m locus.scoring.node2vec --p 1 --q 8                     # -> work/node2vec_p1q8_{vectors.npy,index.json}
.venv/bin/locus sweep     --embeddings work/embeddings_val.parquet  --assoc node2vec --vectors work/node2vec_p1q8 --out work/sweep_val_node2vec.json
.venv/bin/locus eval rank --embeddings work/embeddings_test.parquet --assoc node2vec --vectors work/node2vec_p1q8 --sweep work/sweep_val_node2vec.json --out work/rank_report_node2vec.json
.venv/bin/locus eval swap --val-embeddings work/embeddings_val.parquet --test-embeddings work/embeddings_test.parquet \
    --assoc node2vec --vectors work/node2vec_p1q8 --sweep work/sweep_val_node2vec.json --tag node2vec
```

**Expect** Rank `0.47771` → `0.49462` (`+0.01691`) at lambda 15, and on Swap
a selected lambda of 0: the node2vec signal does not help there.

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
