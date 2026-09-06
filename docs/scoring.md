# Scoring LOCUS with your own model

Three ways in, in order of how much of your code we have to see.

## 1. A parquet file (no code)

The lowest-commitment path: produce a parquet and hand it to `locus verify`.
The schema is in [schema.md](schema.md#the-two-parquet-contracts).

```bash
locus verify --export work/export \
    --embeddings /path/to/your_vectors.parquet \
    --graph      work/cocitation_train.npz
```

The dimension is inferred, not fixed — HAtten runs at 200 dims against
SPECTER2's 768. Vectors are L2-normalised at load, so scale does not matter;
a zero vector is a hard error. A directory of `*.parquet` shards is read in
sorted name order and is equivalent to one file.

The keys you must cover are exactly what `locus build embed-inputs` collects.

If you do not have your own embedder, `locus embed` (`locus.scoring.embed`)
reproduces the pinned production recipe locally on a GPU or a laptop:
`allenai/specter2_base`, CLS pooling, fp16 on CUDA and fp32 elsewhere -- no
AWS credentials, at the cost of `pip install -e ".[gpu]"` for
`torch`/`transformers`. `locus.embed_specter2` is the S3-native sibling this
reproduces: it downloads its model snapshot from S3, stages its input from S3
and uploads its shards to S3, so it is **not runnable outside the deployment
it was written for** and needs `boto3` (`pip install -e ".[cloud]"`) rather
than a `locus` verb of its own. Either produces the parquet above; any
bi-encoder will do, and the schema is the only thing that has to match.

### The paper's other bases

`locus embed` is not SPECTER2-only. `--model` takes any Hugging Face
bi-encoder, and the loader pins a revision per model so a rerun loads the
same weights: `allenai/specter2_base` at `3447645e` and `malteos/scincl` at
`ebc5348d`. A model without a pin loads at the hub's current head with a
warning; pass `--revision <commit>` to make that reproducible.

```bash
locus embed --model malteos/scincl \
    --input work/embed_inputs_test.parquet --output work/embeddings_scincl_test.parquet
```

HAtten is not a Hugging Face model. `python -m locus.experiments.hatten`
embeds the frozen pools with Gu et al.'s hierarchical-attention prefetcher,
and needs three things from their Local-Citation-Recommendation repository,
linked from the ECIR 2022 paper (doi:10.1007/978-3-030-99736-6_19): the arXiv
prefetch checkpoint (`model_batch_1645000.pt`), its vocabulary
(`vocabulary_200dim.pkl`), and the path to their `src/prefetch` directory,
whose pure-torch `model.py` is imported rather than vendored.

```bash
python -m locus.experiments.hatten --split test \
    --ckpt  /path/to/model_batch_1645000.pt \
    --vocab /path/to/vocabulary_200dim.pkl \
    --lcr-src /path/to/Local-Citation-Recommendation/src/prefetch
```

It writes `work/embeddings_hatten_{split}.parquet` at 200 dimensions. Its
query is three paragraphs (the citing paper's title, its abstract, and the
masked context), which is HAtten's own design and strictly more input than
the other bases see; the within-base delta is unaffected, the cross-base
comparison of raw MRR is not like for like.

## 2. The `Scorer` protocol (a few lines of code)

If your model cannot be reduced to one vector per item — a cross-encoder, a
retrieval API, anything with state — implement the protocol instead.

```python
from locus.core.protocols import PoolScorer

class MyScorer:
    def __call__(self, context_id: str, candidate_id: str) -> float:
        ...

    def pool_scores(self, context_id: str, candidates) -> dict[str, float]:
        return {c: self(context_id, c) for c in candidates}
```

`__call__` is the whole obligation; `pool_scores` exists because the lambda
sweep scores the same pool at many lambdas, and doing that one candidate at
a time repeats 554,880 dot products per lambda. Implement it if scoring a
pool together is cheaper than scoring its members separately.

Both are `runtime_checkable`, so `isinstance(obj, PoolScorer)` tells you
whether you have satisfied it.

## 3. The bundled scorers

`locus.scoring.dense.DenseScorer` reads the parquet from layer 1.
`locus.scoring.bm25.BM25Scorer` is the lexical base — `--texts` in place of
`--embeddings` on every eval entry point.

```bash
locus eval rank --texts work/embed_inputs_test.parquet --sweep work/sweep_val_bm25.json
```

Okapi BM25 with Lucene's `log(1 + (N-n+0.5)/(n+0.5))` idf over the *paper*
rows, `k1 = 1.5`, `b = 0.75`, untuned. On test it scores MRR `0.50174` ->
`0.60399` (`+0.10226`), which is a **stronger base than any dense encoder
measured here** -- `hardness.is_easy` suppresses title and author overlap
between window and target, not abstract overlap, so the hard slice is not
lexically neutral.

It is not in `scoring/baselines.py`: those scorers are context-independent by
construction and pinned to the floor by the lemma, which makes them
assertions about the harness. BM25 reads the query, so nothing pins it.

## What a fair comparison requires

Two rules, both of which the harness enforces and neither of which is
optional:

  - **No candidate filtering, anywhere.** The chance floor is `H_k/k` and
    moves with pool size — 0.29290 at k=10, 0.31433 at k=9. Any filter that
    can drop a candidate invalidates the frozen floor for the pools it
    touches, including the abstract-length gates conventional in embedding
    pipelines.
  - **Ties are scored by expectation**, never broken randomly. A scorer that
    ignores the context must land on exactly `H_10/10` on Rank and exactly
    0.5 on Swap. If yours does not, the plumbing is wrong, not the model.
