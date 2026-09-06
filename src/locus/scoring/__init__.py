

def base_scorer(embeddings=None, texts=None):
    """Build the base scorer named by exactly one of --embeddings / --texts.

    Both produce a `PoolScorer` over the same frozen pools, so every eval
    entry point takes either and nothing downstream of here knows which was
    chosen. Refusing both-or-neither here rather than in five argument
    parsers keeps the two flags from drifting apart between them.
    """
    if (embeddings is None) == (texts is None):
        raise SystemExit(
            "pass exactly one of --embeddings (a dense parquet) or --texts "
            "(a (key, kind, text) parquet, for BM25)"
        )
    if texts is not None:
        from locus.scoring.bm25 import BM25Scorer

        return BM25Scorer(texts)
    from locus.scoring.dense import DenseScorer

    return DenseScorer(embeddings)
