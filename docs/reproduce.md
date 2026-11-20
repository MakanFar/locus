# Reproducing the paper

Three tiers. Each states what it needs, how long it takes and **the exact
numbers you should see**, so a failed reproduction is distinguishable from
a slow one.

| Tier | Needs | Time | Produces |
|---|---|---|---|
| 0 | the release bundle (~144 MB) | minutes, laptop | floors, export integrity, and the headline result once embeddings are on disk (see below) |
| 1 | + the upstream corpus, and one GPU-hour *or* Tier 0's precomputed vectors | hours | all four bases on both probes, including BM25 |
| 2 | + corpus-wide vectors (~2 GB) | hours | sequential completion, anchor influence, anchor analysis |

`scripts/reproduce.sh` runs everything below in order, one step at a time:
`scripts/reproduce.sh tier0` for Tier 0, `scripts/reproduce.sh all` for
Tiers 1 and 2, or name individual stages (`--list`). It skips steps whose
output already exists, skips with a reason those whose inputs are missing,
and keeps each step's output in `$LOCUS_WORK_DIR/logs/`; `--dry-run` prints
the commands without running them. The sections below are the same commands
with the numbers each should print.

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
(<https://doi.org/10.5281/zenodo.24383365>, version 0.3.0, CC-BY-4.0) and
verifies every file's sha256 against `bundle_manifest.json` before anything
lands; `--manifest-url` points it somewhere else. The bundle is **flat**:
`locus_rank_test.jsonl`, `locus_swap_test.jsonl`, `locus_anchors_test.jsonl`,
`locus_manifest_test.json` and `cocitation_train.npz` (plus the `val` split)
land directly in `work/release/` — there is no `export/` subdirectory, and
the manifest itself is read from Zenodo, not saved. It does **not** carry `embeddings_test.parquet`: precomputed
embeddings are derived from upstream abstracts and are deliberately kept out
of the core bundle so its DOI survives independently of that separate
deposit: <https://doi.org/10.5281/zenodo.24383583> holds
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
| Rank pools (test) | 54,020 |
| Headline pools | 34,577 |
| Contexts | 104,401 |
| Citing papers | 9,208 |

These counts are for the corrected pools (distractors drawn outside both
anchor routes; see CHANGELOG). The 0.3.0 deposit carries the corrected pools;
the superseded 0.2.0 version (doi:10.5281/zenodo.22260046) declares 55,488
rank pools and 35,701 headline pools and will not match the numbers below.

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
  https://zenodo.org/records/24383583/files/embeddings_test.parquet
```

then:

```bash
.venv/bin/locus verify --export work/release \
    --embeddings work/embeddings_test.parquet \
    --graph      work/release/cocitation_train.npz
```

**Expect:** MRR `0.49672` → `0.62054` (`+0.12382`, [+0.11594, +0.13247]) and
R@1 `0.30307` → `0.45956` (`+0.15648`).

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
(<https://doi.org/10.5281/zenodo.24383583>) if you would rather not run
`locus embed`. **Expect** on Rank `0.49672` → `0.62054` (`+0.12382`, with
alpha 0.75, lambda 3.0 selected on val) and on Swap `0.81602` → `0.84358`
(`+0.02755`, lambda 0.2). Every metric is a mean of per-citing-paper means
(`averaging: "paper"` in the report, over 1,968 papers); the same comparison
averaged over pools is `gate1_pool_level`, `+0.12905` [+0.12222, +0.13601],
whose interval still resamples citing papers; `eval rank` also prints
the coverage strata (Table 9) and the other-location control against the
base on every run.

**Expect** from `build pipeline --split test` the composition the paper's
Table 8 and Section 3 quote: 104,401 contexts from 9,208 citing papers in
55,714 locations; 69,874 contexts with a clustering-route anchor, 70,357 with
a marker-route anchor and 78,334 with either (75.03%; the remaining 24.97%
have no resolvable anchor and never reach the headline set); 91,352 contexts
(87.5%) align their masked window to the raw text, and the inline markers in
those windows resolve to 244,345 in-corpus anchors, with 182,355 unresolvable
(cited work outside the corpus), 28,148 implausible numbers and 12,831 self
references -- 52.2 / 39.0 / 6.0 / 2.7% of the 467,679 markers; 54,020 rank
pools, 27,905 swap pairs, a hard fraction of 78.21% and 34,577 headline
pools. `build graph` counts 374,691 papers and 3,436,198 directed non-zeros
with zero overlap against the 20,845 held-out papers.

**`--split val` exits 1 and that is expected.** It fails one gate —
alignment failure 17.4% against a 15% threshold, versus 12.5% on test. Every
artefact is still written; the gate verdict comes after every `pickle.dump`.
The failures never reach a pool (all 19,585 are dropped by
`build_rank_items`), val and test share zero citing papers and zero
contexts, and the surviving val headline set is shaped like test's — 30,177
items across 1,899 papers, hard fraction 78.44% against test's 78.21%. The
rest of Table 8's val column is 112,779 contexts from 11,637 citing papers in
62,466 locations, 72.23% of them anchored, 47,946 rank pools and 33,203 swap
pairs. The gate was left as it is because it reports something true.

BM25 lives at this tier, not Tier 0, because it reads
`embed_inputs_{split}.parquet`, which carries upstream titles, abstracts and
context windows and is therefore not redistributable:

```bash
.venv/bin/locus sweep     --texts work/embed_inputs_val.parquet  --out work/sweep_val_bm25.json
.venv/bin/locus eval rank --texts work/embed_inputs_test.parquet --sweep work/sweep_val_bm25.json --out work/rank_report_bm25.json
```

**Expect** base MRR `0.52212` → `0.62579` (`+0.10366`), and on Swap
(`--val-texts`/`--test-texts` in place of the embeddings flags, `--tag bm25`)
`0.83404` → `0.85884` (`+0.02480`, lambda 0.1).

### The remaining rows of Tables 1, 2 and 10

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

**Expect** Rank SciNCL `0.49028` → `0.61463` and HAtten `0.48009` → `0.59699`;
Swap SciNCL `0.80089` → `0.83369` and HAtten `0.83073` → `0.85313`.

The controls table (Table 10), all on SPECTER2:

```bash
.venv/bin/locus eval rank --embeddings work/embeddings_test.parquet --slice easy --out work/rank_report_easy.json
.venv/bin/python -m locus.eval.controls --embeddings work/embeddings_test.parquet
.venv/bin/locus sweep     --embeddings work/embeddings_val.parquet  --assoc count --out work/sweep_val_count.json
.venv/bin/locus eval rank --embeddings work/embeddings_test.parquet --assoc count --sweep work/sweep_val_count.json --out work/rank_report_count.json
```

**Anchor route** (Table 7 of the appendix). The same headline pools,
rescored with anchors from one inference route at a time:

```bash
.venv/bin/locus eval rank --embeddings work/embeddings_test.parquet --anchor-route marker     --out work/rank_report_route_marker.json
.venv/bin/locus eval rank --embeddings work/embeddings_test.parquet --anchor-route clustering --out work/rank_report_route_clustering.json
```

**Expect** `+0.12216` with markers only (34,505 of the 34,577 pools keep an
anchor) and `+0.13005` with clustering only (31,121 keep one), against
`+0.12382` for the union; a pool with no anchor under a route scores as the
base does, and `gate1_on_anchored` restricts to the pools that have one
(`+0.12254` and `+0.14754`).

**Anchor proximity.** How the reranker does when given only the anchors that
share a citation group with the target, only those elsewhere in its sentence,
or only those in another sentence of the window:

```bash
.venv/bin/locus exp proximity --embeddings work/embeddings_test.parquet --out work/proximity_specter2.json
```

**Expect** `+0.18958` (same group, 22,218 pools), `+0.07905` (same sentence,
14,432), `+0.00980` (other sentence, 14,724) and `+0.08619` (outside the
window, 1,954); the closest anchor a headline pool has is in the target's
citation group for 64.3% of pools, elsewhere in its sentence for 21.3%, in
another sentence for 14.3% and outside the window for 0.2%; and on the
22,218 pools with a same-group anchor, adding the same-sentence tier lowers
the gain to `+0.17940` and adding the other-sentence tier to `+0.16512`. The
tiers are regular expressions over the masked window, not annotations.

**Clustering threshold.** The benchmark is frozen at theta = 0.5. To see what
the headline does at another threshold, everything downstream of the
clustering is rebuilt in its own work directory -- index, pools, hardness,
the train-only graph -- and lambda is re-selected on a validation split
rebuilt at the same threshold (`scripts/reproduce.sh theta` runs all of it):

```bash
for T in 0.3 0.4 0.6 0.7; do
  export LOCUS_WORK_DIR=work/theta_$T
  .venv/bin/locus build pipeline --split test --theta $T
  .venv/bin/locus build pipeline --split val  --theta $T
  .venv/bin/locus build graph --theta $T
  for S in test val; do
    .venv/bin/locus build embed-inputs --split $S
    .venv/bin/locus embed --input work/theta_$T/embed_inputs_$S.parquet --output work/theta_$T/embeddings_$S.parquet \
        --reuse work/embeddings_$S.parquet        # embeds only the rows the published vectors lack
  done
  .venv/bin/locus sweep --embeddings work/theta_$T/embeddings_val.parquet --graph work/theta_$T/cocitation_train.npz --out work/theta_$T/sweep_val.json
done
export LOCUS_WORK_DIR=work
.venv/bin/locus exp theta --dir 0.3=work/theta_0.3 --dir 0.4=work/theta_0.4 --dir 0.5=work \
    --dir 0.6=work/theta_0.6 --dir 0.7=work/theta_0.7 \
    --fixed-graph work/cocitation_train.npz --fixed-sweep work/sweep_val.json --out work/theta_sensitivity.json
```

The validation build exits 1 at every threshold on the same alignment gate it
fails at 0.5; its artefacts are written regardless. Changing the threshold
changes which contexts are eligible, which nine distractors each pool holds
and which anchors each context has, so `exp theta` reports four readings per
threshold: everything rebuilt; the contexts eligible at every threshold; the
contexts whose ten candidates are identical at every threshold; and the
theta = 0.5 pools held fixed and rescored with each threshold's anchors.
This is a sensitivity analysis and must not be used to choose theta.

**Expect** (Table 5) 48,747 / 52,177 / 55,714 / 59,391 / 62,839 locations at
theta 0.3 to 0.7, with 5.1 / 10.2 / 17.0 / 25.2 / 33.1% of marker anchors
outside their context's cluster, 76.0 to 73.9% of contexts anchored, a mean
anchor-set size of 4.27 to 3.91, and validation selecting (alpha, lambda) =
(0.75, 4), (0.75, 3), (0.75, 3), (0.75, 4) and (0.5, 3). Then (Table 6) a
full-rebuild delta of `+0.12235`, `+0.12179`, `+0.12382`,
`+0.12526` and `+0.12512` (34,010 to 34,667 headline pools); 33,916 contexts
eligible at every threshold, on which the delta is `+0.12244`, `+0.12246`,
`+0.12432`, `+0.12629` and `+0.12641`; and, with the
theta = 0.5 pools held fixed and only the anchors swapped (31,142 pools),
`+0.12699` to `+0.12749`. The contexts whose pool is identical at every
threshold (29,345) have identical anchors too for 29,276 of them, so that
reading varies only the graph: `+0.12727` to `+0.13078`. About an hour,
most of it the four graph builds and the `--reuse` embeddings.

**The annotation sample.** Locations and anchors are inferred, so
`docs/annotation_protocol.md` describes a hand-label check by two
annotators. This draws the 400 pairs they label and a 40-pair pilot,
stratified by what the pipeline decided, and writes one sheet per annotator
(the same items in a different order, with nothing about the pipeline's
decision), a key, and a manifest with each cell's population:

```bash
.venv/bin/locus exp annotation-sample --out work/annotation
```

**Expect** 1,722,838 ordered pairs in the test split and every cell filled:
60 where both routes agree, 80 marker-only, 40 and 60 clustering-only
(aligned and unaligned), 60 joined only transitively, and 60 / 20 / 20
non-anchors by window overlap. The sheets contain corpus text and stay in
the work directory; the key and manifest hold identifiers and counts only,
and the manifest is committed as `reports/annotation_manifest.json`.

**Scoring the labels.** The two readers' labels and the adjudicated
consensus are committed with the key, ids only (`reports/annotation_key.csv`,
`reports/annotation_labels.csv`, and the pilot alongside). Scoring them needs
no corpus; the last block, which rescores the headline pools the labels
touch, needs the test embeddings, the graph and the sweep selection:

```bash
.venv/bin/locus exp annotation-score --key reports/annotation_key.csv --labels reports/annotation_labels.csv --pilot-key reports/annotation_pilot_key.csv --pilot-labels reports/annotation_pilot_labels.csv --build-report reports/build_report_test.json --embeddings work/embeddings_test.parquet --out work/annotation_score.json
```

**Expect** raw agreement `0.860`, kappa `0.792` on the four-way label
(`0.892` for G against the rest, `0.860` for G-or-S against the rest), 56
items adjudicated, and a pilot at `0.750` / `0.622`. Post-stratified, site =
sentence: clustering precision `0.852` [0.833, 0.870], marker `0.759`,
union `0.754`; site = group: `0.553`, `0.472`, `0.469`. Recall under the
group definition `0.998` / `0.909` / `1.000`; under the sentence definition
`0.935` [0.909, 0.950] / `0.889` / `0.977`. The regex tier matches
the consensus on 182 of the 186 items where it assigns one. On the 102
headline pools an annotated false positive touches, the gain is `+0.15474`
with the pipeline's anchors and `+0.15789` with that anchor removed.
Seconds without `--embeddings`; about a minute with.

The appendix compares the hard and easy gains, and a ratio or difference of
two separately bootstrapped means has no interval of its own, so the
comparison is scored once more with one paper-level bootstrap over both
slices:

```bash
.venv/bin/python scripts/hard_easy_contrast.py --embeddings work/embeddings_test.parquet --out work/hard_easy_contrast_specter2.json
```

**Expect** hard `+0.12382` [+0.11594, +0.13247] against easy `+0.02183`
[+0.00974, +0.03324]; difference `+0.10198` [+0.08937, +0.11440]; ratio
`5.67` [3.75, 11.86].

**Expect** easy slice `0.68017` → `0.70200` (`+0.02183`); random anchors
`0.49646` and degree-matched `0.49260` against the `0.49672` base; raw
co-counts `+0.11083` at lambda 5; and, printed by every `eval rank` run, the
other-location control `0.39732` (`-0.09940` against the base).

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

**Expect** base `0.55425` → `0.63291` (`+0.07866`).

**Expected wall clock and memory:** the two `build pipeline` runs are under
a minute each on a fast disk (~5 minutes on a slow one) and need ~8 GB of
RAM; `build graph` about 5 minutes; every sweep, rank, swap
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
.venv/bin/locus exp completion --embeddings work/embeddings_expb_test_mm --mode revealed --standardize none --out work/expb_report_revealed.json
.venv/bin/locus exp completion --embeddings work/embeddings_expb_test_mm --mode selfseed --standardize none --out work/expb_report_selfseed.json
.venv/bin/locus exp influence  --embeddings work/embeddings_test.parquet --sim-store work/embeddings_expb_test_mm --dump work/influence_rows_specter2.pkl
.venv/bin/locus exp anchors --tag specter2
```

**Expect** 22,274 locations per completion run (about 7-30 minutes each);
with a revealed seed Recall@20 `0.56288` → `0.71576` (`+0.15288`), which
is Table 3 setting A, and self-seeded `0.26383` → `0.38557` (`+0.12174`),
setting B. In the self-seeded run the first pick is correct at 2,091 of the
22,274 locations per seed (9.39%), after which Recall@20 is `0.72308` against
`0.33262` after a wrong first pick, and the semantic arm's Recall@1 is `0.04264`.
The influence pass makes 136,888 interventions, categorised 11,556
informative (8.4%, mean rank influence `+0.41665`) / 43,415 redundant (31.7%) /
81,917 non-co-cited (59.8%, `-0.02962`); mean rank influence by anchor-set size
is `+0.08397` at one anchor, `+0.03016` at two, `+0.01576` at three and
`+0.00027` at nine; and `anchor_map_specter2.json` plus its pgfplots table
is Figure 3, whose top PPMI band runs from `+0.21` to `+0.29` across the
cosine octiles.

`exp completion` prints two tables. The first is Recall@K counting the
revealed seed, which is what the published completion numbers are. The
second scores the same runs over `G(l)` minus that seed -- the members the
system still has to find -- because the first hands every arm `1/|G(l)|` for
free and 56.6% of locations have `|G(l)| = 2`. The paper's sequential-completion numbers use the raw cosine
as the base (`--standardize none`); the default `--standardize zscore`
z-scores the base over the location's candidate list, which is Eq. 1's scale
and gives slightly larger gains (self-seeded Recall@20 `0.39964`). **Expect**,
on the remaining members with a revealed seed, Recall@1 `0.04753` →
`0.13189` (`+0.08436`), Recall@2 `0.07734` → `0.20591` (`+0.12857`), Recall@5
`0.13354` → `0.32891` (`+0.19537`) and Recall@20 `0.26497` → `0.51845`
(`+0.25348`) -- Table 4's oracle-prefetch rows -- and `1.00000`
reachability, since the oracle prefetch guarantees the gold is a candidate.

Corpus-wide retrieval replaces the oracle prefetch with the top 2,000
papers by cosine over the whole store; the retrieval step writes the task
file the completion runs then take through `--task`:

```bash
.venv/bin/python -m locus.scoring.retrieve --store work/embeddings_expb_test_mm --k 2000   # -> expb_task_test_corpuswide.jsonl, retrieve_report_test.json
.venv/bin/locus exp completion --embeddings work/embeddings_expb_test_mm --mode selfseed --standardize none \
    --task work/expb_task_test_corpuswide.jsonl --out work/expb_report_corpuswide.json
.venv/bin/locus exp completion --embeddings work/embeddings_expb_test_mm --mode revealed --standardize none \
    --task work/expb_task_test_corpuswide.jsonl --out work/expb_report_revealed_corpuswide.json
```

**Expect** first-stage recall over 67,492 gold citations of `0.31333` at
k=100 and `0.68381` at k=2000; with a revealed seed Recall@20 `0.52177` →
`0.62719` (`+0.10543`), which is Table 3 setting C; self-seeded `0.19377`
→ `0.26872` (`+0.07495`), setting D; and, on the remaining members with a
revealed seed, Recall@1 `0.03468` → `0.08943` (`+0.05475`), Recall@2 `0.05846`
→ `0.14427` (`+0.08581`), Recall@5 `0.10043` → `0.23085` and Recall@20
`0.19313` → `0.36834` (`+0.17522`) -- Table 4's corpus-wide rows -- with a
reachability of `0.69344`: corpus-wide the gold is not guaranteed to be
retrieved at all, and that ceiling binds both arms equally. (The 0.684 in
the paper's Section 5.3 is the same retrieval step's recall pooled over all
67,492 gold citations; the 0.693 in Table 4's caption is the per-location
mean over the members left after the revealed seed.)

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

**Expect** Rank `0.49672` → `0.54192` (`+0.04519`) at lambda 30, and on Swap
a selected lambda of 0: the node2vec signal does not help there. The other
four `q` values (`--q 0.5`, `1`, `2`, `4`) score lower on validation
(`0.53477`, `0.53772`, `0.53866`, `0.54046` against `0.54134` at `q = 8`) and
also select lambda 0 on Swap. The appendix's node2vec-in-completion sentence
is `locus exp completion --assoc node2vec --vectors work/node2vec_p1q8
--sweep work/sweep_val_node2vec.json --standardize none` in the three
settings: **expect** Recall@20 deltas of `+0.00068` (revealed seed, interval
includes zero), `-0.07285` (self-seeded) and `-0.05385` (corpus-wide).

## If a number does not match

Run `locus doctor` first — it reports Python version, which optional extras
import, what `LOCUS_DATA_DIR` and `LOCUS_WORK_DIR` point at, and whether a
bundle is present. It always exits 0, so it is safe to run when something
else has already failed.

See also [schema.md](schema.md) for the artefact layout, [scoring.md](scoring.md)
for the scorer protocol, and [custom-corpus.md](custom-corpus.md) for running
LOCUS against a different corpus.

## What the paper reports that the harness does not produce

Figure 2 and the coherence numbers in Sections 1.3 and 5.1 -- the mean
pairwise cosine similarity of SPECTER2's top-|A| retrieved sets against
human citation sets (0.944 / 0.933 / 0.928 and the +0.014 gap with its
section split) and the two worked examples' percentiles -- were produced
outside this harness and have no report under `reports/`. Every other number
in the paper is listed in `reports/paper_map.json`.

## What the harness can run that the paper does not report

These exist and are tested, but no number in the paper comes from them:
`--agg masked_mean` / `--agg max` on `locus sweep` and `locus eval rank`
(alternative anchor-set aggregators, each with its own lambda sweep), the
node2vec null control (`python -m locus.scoring.node2vec --null-control`),
and `locus exp completion --standardize zscore` (Eq. 1's z-scored base in
sequential completion; the paper reports the raw-cosine base). Treat their
outputs as ablations, not as reproductions.
