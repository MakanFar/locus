# Schema

The normative contracts. Four files carry data across the boundary between
LOCUS and everything else: two you bring in, one we publish, and two parquet
formats that carry vectors and text.

Field tables in this document are pinned to the code by
`tests/test_docs_schema.py` — a field renamed in `locus/data/records.py` and
not here fails the suite.

## The corpus you bring in

    <root>/
        contexts.jsonl                 one JSON object per line
        papers.jsonl                   one JSON object per line
        splits/train.json              JSON array of context_id strings
        splits/val.json                JSON array of context_id strings
        splits/test.json               JSON array of context_id strings

Validate it with `locus ingest <root>`, which exits 0 iff there are no
problems and reports every violation rather than stopping at the first.

### `contexts.jsonl`

One row per citation context.

| Field | Type | Meaning |
|---|---|---|
| `context_id` | str | unique across the file |
| `citing_id` | str | the paper this window was taken from |
| `refid` | str | the paper this context cites; must appear in `papers.jsonl` |
| `raw` | str | the window verbatim, citation markers intact |
| `masked_text` | str | the same window with **exactly one** `TARGETCIT`, and `OTHERCIT` for every other marker |

### `papers.jsonl`

One row per paper. Every `refid` named in `contexts.jsonl` must be here.

| Field | Type | Meaning |
|---|---|---|
| `id` | str | matches `citing_id`/`refid` in `contexts.jsonl` |
| `title` | str | may be empty |
| `abstract` | str | may be empty |
| `authors` | list | **optional**; a JSON array, first author first |

**`raw` is required, and it is not decorative.** It drives both of the ways a
context gets an anchor set `A(l)`, which are not nested and both feed the
headline slice:

  - *location clustering.* `core.locations.cluster` labels locations by word-shingle
    Jaccard over `raw`. Contexts that are the same citation site get the same label.
  - *marker alignment.* `core.alignment.markers_with_numbers(raw, masked_text)` walks
    the two strings in step to recover which bibliography number each marker
    carried. On Gu et al.'s test split that route supplies 70,357 of 78,334
    anchored contexts.

A corpus whose `raw` does not correspond to its `masked_text` still builds. It
just loses the second route silently -- no error, a smaller anchor set, a
smaller headline slice. That is why alignability is a validator check and not
only a documented requirement, and why it is a corpus-level *rate* rather than
a per-row error: a few windows cut mid-marker is normal (Gu et al.'s own test
split sits at 2.33%), most of them failing means `raw` and `masked_text` are
not the same window at all.

**`authors` is optional, and omitting it is measurable, not free.**
`core.hardness.is_easy` calls a context easy when the first author's surname is
in the window, or when at least `HARDNESS_TAU` of the title's content words
are. Without `authors` only the second route survives. On Gu et al.'s test
split, over the same 55,488 rank items: hard 43,372 (78.16%) with authors,
45,611 (82.20%) without, and the headline slice moves 35,701 -> 37,208. A
corpus without authors is a valid LOCUS corpus; it is not the corpus the
published hard fraction was measured on.

## The corpus Gu et al. provides

The Local Citation Recommendation arXiv split (Gu et al., ECIR 2022) is what
`$LOCUS_DATA_DIR` must point at when no `--corpus` is given -- five files in
one directory, not the JSONL contract above:

    contexts.json    JSON object, context_id -> {citing_id, refid, raw,
                     masked_text}: the citation window verbatim and with its
                     own marker masked, the paper it is in, the reference it
                     cites
    papers.json      JSON object, paper_id -> {title, abstract}
    train.json       JSON array of records carrying `context_id`
    val.json         the same, for the split that selects lambda
    test.json        the same, for the split that reports the headline

`locus.data.streaming` parses these incrementally rather than with
`json.load`, so the two object files never have to fit in memory at once. That
puts one requirement on `contexts.json` beyond its shape: its entries must be
**contiguous by `citing_id`**, since the loader closes a citing paper as soon
as a different one appears. It holds for all 3,205,210 records of the upstream
file, and `load_index` raises rather than silently splitting a paper if it
ever stops holding.

## The frozen artifacts

```bash
.venv/bin/locus build pipeline --split test   # the headline split
.venv/bin/locus build pipeline --split val    # selects lambda
```

Writes, all under `work/` and all suffixed with `{split}`: `index_{split}.pkl` (the frozen
per-citing-paper index), `rank_items_{split}.pkl` and `swap_pairs_{split}.pkl` (the frozen
LOCUS-Rank pools and LOCUS-Swap pairs), `hard_ids_{split}.pkl` and
`headline_ids_{split}.pkl` (sorted lists of context ids -- deterministic, not sets), and
`build_report_{split}.json`. Exit code 0 iff every gate passes.

The split suffix is what keeps lambda selection honest: `val` and `test` artefacts sit
side by side and neither can overwrite the other, so a lambda swept on val can never be
swept on the split that reports the headline. `train` is deliberately not a choice -- it
has no pools and no gates, and reaches the graph through `locus.build.graph`, which
streams it rather than indexing it. For why `val` exits 1 on its alignment gate and is
still used to select lambda, see the Tier 1 note in [reproduce.md](reproduce.md).

`build_report_{split}.json` records, alongside `split` and the frozen config: `contexts`, `citing_papers`,
`alignment_success`, `alignment_failure`, `alignment_parse_failure`,
`alignment_validation_failure`, `num_locations`, `contexts_per_location`,
`anchor_distribution`, `anchor_ge1_clustering`, `anchor_ge1_marker`, `anchor_ge1_union`,
`marker_anchors_unresolvable`, `marker_anchors_implausible`, `marker_anchors_total`,
`clustering_disagreement`, `pool_drop_reasons`, `rank_items`, `swap_pair_count`, `hard`,
`headline`, `null_controls`, `failures`.

A context's anchor set A(l) has two evidence routes that are **not** nested: the
clustering route (`anchors` -- another context at this location targets it) and the
marker route (`rec.marker_anchors` -- an OTHERCIT marker in this window resolves to it).
`anchor_distribution`, the `|A|>=1` gate and headline-set membership all use
`anchors_union`, the union of both routes; `anchor_ge1_clustering`, `anchor_ge1_marker`
and `anchor_ge1_union` are reported side by side so the divergence between routes stays
visible.

`clustering_disagreement` is the THETA diagnostic: a context's own OTHERCIT markers sit inside
its own window, so a correct cluster must contain them. Any that fall outside mean the cluster
was split too finely. It needs no ground truth, so sweep THETA against it directly --
against the rate (`clustering_disagreement` / `marker_anchors_total`), since the raw count
also moves with how many marker anchors were recovered.

## Gates

| Gate | Threshold | Scope |
|---|---|---|
| Alignment failure rate | <= 15% (expect 8-11%) | any corpus |
| Anchor coverage \|A\|>=1 (union) | >= 60% | any corpus |
| Hard fraction | 40-85% | Gu et al.'s test split |
| Headline set | >= 30,000 | Gu et al.'s test split |
| Null controls on Swap | **exactly** 0.5, with a zero-width bootstrap interval | any corpus |

The thresholds live in `locus.config` and are pinned by
`tests/equivalence/test_frozen_config.py`, so this table cannot drift from the
code. The last two describe one corpus rather than the benchmark, which is
why `--corpus` reports them instead of enforcing them -- see
[custom-corpus.md](custom-corpus.md).

The null controls are not baselines. By the lemma, a scorer that ignores the query
context has a bit-exactly zero swap delta, so with fractional tie scoring it must
return `0.5` exactly. Anything else is a bug in pool construction or score plumbing.

## The export we publish

```bash
.venv/bin/locus build export --split test    # writes work/export/
.venv/bin/locus build export --split test --verify
```

The frozen artefacts are pickles, which is right for us and wrong for anyone
else: a pickle is version-coupled to the dataclass that wrote it, unpickling
executes code, and `A(l)` is not in the pool artefacts at all -- it is
recomputed on demand from `index_{split}.pkl`, which is 103 MB and carries the
raw context windows. So reproducing a rescoring result currently means shipping
the upstream corpus text to get at a set of id strings.

`export.py` writes the same information as JSONL carrying **no text of any
kind** -- only ids, integers and booleans. That is what makes it
redistributable: the identifiers are ours to publish, the abstracts and context
windows are Gu et al.'s.

| File | test rows | Contents |
|---|---:|---|
| `locus_rank_{split}.jsonl` | 55,488 | `context_id, citing_id, gold, candidates[10], hard, headline` |
| `locus_swap_{split}.jsonl` | 27,905 | `citing_id, ctx_a, ctx_b, gold_a, gold_b` |
| `locus_anchors_{split}.jsonl` | 104,401 | `context_id, citing_id, location, gold, alignment_ok, anchors, anchors_clustering, anchors_marker` |
| `locus_manifest_{split}.json` | -- | counts, frozen config, exact floors, sha256 per file |

46 MB for the whole test split. `candidates` keeps the frozen shuffled order:
sorting it would change no metric, since scoring is keyed by candidate, but it
would silently publish a different artefact from the one the paper measured.

**Anchors are a separate file, not a column on the rank rows.** Duplicating
them would make `locus_rank` self-sufficient, but the anchor sets are also
needed for contexts with no rank pool -- `build_rank_items` drops alignment
failures and whole papers with too few distractors -- and sequential completion keys on
locations rather than rank items. One row per context is the only shape that
serves all three, and the join is a dict lookup on `context_id`. That file also
carries each context's own `gold`, which is what lets a consumer rebuild the
popularity null: `locus.build.pipeline` counts targets over every context in the index,
not over rank-item golds (step 6), and scoping it to rank items collapses
popularity into the constant scorer on a third of the swap pairs.

`--verify` re-reads the export and checks three things: sha256 per file,
round-trip equality against the pickles, and the null controls **recomputed
from the exported files alone**. The last is the one that matters, because it
is the claim a consumer who never sees our pickles depends on.

### The floor comes from the data, not from `config.POOL_SIZE`

`H_k/k` moves with `k` -- 0.29290 at 10, 0.31433 at 9 -- and the consumer
checks it by equality, so the manifest must state the floor of the file they
hold rather than of our configuration. `pool_size_of` derives `k` from the
exported pools and refuses a set with mixed sizes: two pool sizes means two
floors, so there is none to publish.

**One asymmetry worth knowing about.** Lemma 1 pins every *pool* to the floor,
and that is what `--verify` gates on. Their paper-level *mean* is a weaker
check, not a stronger one: `math.fsum([h]*n)/n` is not bit-identical to `h` for
every `n`, because the final division rounds. Measured over group sizes 1-2000
it differs for 53 of them at `H_10/10`, 83 at `H_9/9` and 423 at `H_5/5`. So a
mismatch on the aggregate alone is a fact about the aggregator, while a
mismatch on any single pool is a defect in the benchmark, and only the second
is a failure. (`locus.build.pipeline` step 6 gates on both and passes today because no
citing paper on either split happens to have one of the unlucky counts.)

### `locus_manifest_{split}.json`

```json
{
  "schema": 1,
  "split": "test",
  "source": {"dataset": "...", "note": "..."},
  "config": {"pool_size": 10, "swap_k": 5, "shingle_n": 8,
             "theta": 0.5, "hardness_tau": 0.5, "seed": 0},
  "floors": {"pool_size": 10,
             "rank_constant_mrr": "0.2928968253968254",
             "swap_context_independent_accuracy": "0.5"},
  "counts": {"rank_items": 55488, "swap_pairs": 27905, "contexts": 104401,
             "citing_papers": 9208, "hard": 43372, "headline": 35701},
  "files": {"locus_rank_test.jsonl": {"rows": 55488, "sha256": "..."}}
}
```

Floors are strings, not floats: they are checked by exact equality and a
JSON float would invite a reader to parse and re-round them.

## The two parquet contracts

Every eval entry point takes a parquet of `(key, kind, embedding)` and infers
the dimension from it. That parquet is the whole contract between a model and
this benchmark:

### `(key, kind, embedding)` — for embeddings

| Column | Type | Meaning |
|---|---|---|
| `key` | `str` | a `context_id` when `kind` is `"context"`, a paper id when it is `"paper"` |
| `kind` | `str` | `"context"` or `"paper"` |
| `embedding` | `Array(Float32, dim)` | the vector; one width for the whole file |

`embedding` must be a polars `Array` (an Arrow fixed-size list), not a
variable-length `List`. A `List` column is refused by name, with a message
saying which column, what dtype it has and what dtype it needs -- it is the
most common way to get this wrong, because building the column from a Python
list of vectors makes polars infer `List` rather than `Array`. Use
`pl.Series(matrix, dtype=pl.Array(pl.Float32, dim))`.

`Array(Float64, dim)` is accepted and **kept** at float64: vectors are stored
in the dtype they arrived in, and each dot product is computed in float32.
Vectors are L2-normalised at load, so their scale does not matter, but a zero
vector is a hard error. A
directory of `*.parquet` shards is read in sorted name order and is equivalent
to one file. The keys you need to cover are exactly what
`locus.build.embed_inputs` collects.

### `(key, kind, text)` — for BM25

| Column | Type | Meaning |
|---|---|---|
| `key` | `str` | a `context_id` when `kind` is `"context"`, a paper id when it is `"paper"` |
| `kind` | `str` | `"context"` or `"paper"` |
| `text` | `str` | the masked window for a context; `"{title} [SEP] {abstract}"` for a paper |

`locus build embed-inputs` writes this file, and `--texts` reads it. It
carries upstream text and is therefore **not redistributable**; see NOTICE.
