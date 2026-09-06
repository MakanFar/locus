"""Score LOCUS with HAtten, the Local Citation Recommendation prefetcher.

HAtten is a bi-encoder like the others, so it produces the same embeddings
parquet -- but at 200 dimensions, over a hierarchical paragraph encoder with
GloVe-initialised word embeddings rather than a BERT stack.

The architecture and its checkpoint are third-party (Gu et al., the
Local-Citation-Recommendation repo). model.py there is pure torch and is
imported from a path given by --lcr-src rather than vendored into this repo.
datautils.py is NOT imported: it pulls nltk at module scope, and rankers.py
additionally pulls matplotlib, faiss and sent2vec. The three pieces of it
that inference actually needs are tiny and reproduced here against the
originals:

    SentenceTokenizer.tokenize   datautils.py:51   `return sen.lower()`
    Vocab.sent2seq               datautils.py:75   lowercase, split, keep
                                                   in-vocab ids, pad/truncate
    encode_document              datautils.py:168  truncate to max_doc_len,
                                                   pad with empty paragraphs,
                                                   mask = 1 where empty

**The query is not the context alone.** test.py:98-100 builds a query from
three paragraphs -- the CITING paper's title, the citing paper's abstract,
and the masked context -- while a candidate is two: its title and abstract.
So HAtten sees strictly more than the SPECTER2 base does, which is its own
design and is used faithfully here, but it makes a cross-base comparison of
raw base MRR a comparison of unequal inputs. The within-base delta that the
plan headlines is unaffected.

Usage:
    .venv/bin/python -m locus.experiments.hatten --split test \
        --ckpt .../model_batch_1645000.pt --vocab .../vocabulary_200dim.pkl \
        --lcr-src /path/to/Local-Citation-Recommendation/src/prefetch
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl

from locus import config
from locus.build.embed_inputs import collect_keys
from locus.data.corpus import load_index, load_papers

EMBED_DIM = 200
MAX_SEQ_LEN = 512
MAX_DOC_LEN = 3
NUM_HEADS = 8
HIDDEN_DIM = 1024
N_PARA_TYPES = 100
NUM_ENC_LAYERS = 1
# datautils.PrefetchDataset defaults, and test.py / compute_papers_embedding.py
TITLE_LABEL = 0
ABSTRACT_LABEL = 1
CONTEXT_LABEL = 3
PADDING_LABEL = 10


class Vocab:
    """datautils.Vocab, minus the nltk import its tokenizer never uses."""

    def __init__(self, words: list[str]):
        self.word_to_index = {w: i for i, w in enumerate(words)}
        self.pad_index = self.word_to_index["<pad>"]

    def sent2seq(self, sent: str, max_len: int) -> list[int]:
        seq = [self.word_to_index[w] for w in sent.lower().split()
               if w in self.word_to_index]
        if len(seq) >= max_len:
            return seq[:max_len]
        return seq + [self.pad_index] * (max_len - len(seq))


def encode_document(vocab: Vocab, paragraphs: list[tuple[str, int]]):
    """datautils.encode_document: pad to max_doc_len, mask marks EMPTY paragraphs."""
    doc = list(paragraphs[:MAX_DOC_LEN])
    doc += [("", PADDING_LABEL)] * (MAX_DOC_LEN - len(doc))
    seqs, types, masks = [], [], []
    for text, para_type in doc:
        seqs.append(vocab.sent2seq(text, MAX_SEQ_LEN))
        types.append(para_type)
        masks.append(1 if text.strip() == "" else 0)
    return seqs, types, masks


def build_rows(split: str) -> list[dict]:
    """Paper and context documents in the paragraph form HAtten expects."""
    work = config.WORK_DIR
    with open(work / f"rank_items_{split}.pkl", "rb") as f:
        rank_items = pickle.load(f)
    with open(work / f"swap_pairs_{split}.pkl", "rb") as f:
        swap_pairs = pickle.load(f)
    paper_ids, context_ids = collect_keys(rank_items, swap_pairs)
    index = load_index(work / f"index_{split}.pkl")
    by_ctx = {r.context_id: r for recs in index.values() for r in recs}

    # A query needs its CITING paper's title and abstract, which is a set of
    # papers disjoint from the candidates -- load both in one pass.
    citing_ids = {by_ctx[c].citing_id for c in context_ids}
    papers = load_papers(paper_ids | citing_ids)

    rows = [
        {"key": pid, "kind": "paper", "paragraphs": [
            (papers[pid]["title"], TITLE_LABEL),
            (papers[pid]["abstract"], ABSTRACT_LABEL),
        ]}
        for pid in sorted(paper_ids)
    ]
    for cid in sorted(context_ids):
        rec = by_ctx[cid]
        citing = papers[rec.citing_id]
        rows.append({"key": cid, "kind": "context", "paragraphs": [
            (citing["title"], TITLE_LABEL),
            (citing["abstract"], ABSTRACT_LABEL),
            (rec.masked, CONTEXT_LABEL),
        ]})
    return rows


def load_encoder(ckpt: Path, vocab: Vocab, vocab_size: int, lcr_src: Path,
                 device: str):
    import torch

    sys.path.insert(0, str(lcr_src))
    from model import DocumentEncoder  # third-party, pure torch

    # pad_index comes from the vocabulary, which puts <pad> at 1 -- <eos> is 0.
    # AddMask masks exactly the positions equal to pad_index, so hardcoding 0
    # would mask every <eos> and leave real padding attended to.
    encoder = DocumentEncoder(
        EMBED_DIM, NUM_HEADS, HIDDEN_DIM, vocab_size, MAX_SEQ_LEN, MAX_DOC_LEN,
        pad_index=vocab.pad_index, n_para_types=N_PARA_TYPES,
        pretrained_word_embedding=None, num_enc_layers=NUM_ENC_LAYERS,
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    encoder.load_state_dict(state["document_encoder"])
    return encoder.eval().to(device)


def embed(rows, vocab: Vocab, encoder, device: str, batch_size: int) -> np.ndarray:
    import torch

    out = np.empty((len(rows), EMBED_DIM), dtype=np.float32)
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        enc = [encode_document(vocab, r["paragraphs"]) for r in chunk]
        seqs, types, masks = (list(x) for x in zip(*enc, strict=True))
        s = torch.tensor(np.asarray(seqs), dtype=torch.long, device=device)
        t = torch.tensor(np.asarray(types), dtype=torch.long, device=device)
        m = torch.tensor(np.asarray(masks) == 1, device=device)
        with torch.inference_mode():
            out[start : start + len(chunk)] = (
                encoder(s, t, m).float().cpu().numpy()
            )
        if start % (batch_size * 100) == 0:
            print(f"  {start}/{len(rows)}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("test", "val"), required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--lcr-src", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    with open(args.vocab, "rb") as f:
        words = pickle.load(f)
    vocab = Vocab(words)
    print(f"device={device} vocab={len(words)}")

    rows = build_rows(args.split)
    encoder = load_encoder(args.ckpt, vocab, len(words), args.lcr_src, device)
    vectors = embed(rows, vocab, encoder, device, args.batch_size)

    norms = np.linalg.norm(vectors, axis=1)
    if not np.all(norms > 0):
        raise SystemExit(
            f"{int((norms == 0).sum())} HAtten vectors have zero norm; a "
            "document whose every token is out of vocabulary encodes to one"
        )

    out = config.WORK_DIR / f"embeddings_hatten_{args.split}.parquet"
    pl.DataFrame({
        "key": [r["key"] for r in rows], "kind": [r["kind"] for r in rows],
    }).with_columns(
        embedding=pl.Series(list(vectors), dtype=pl.Array(pl.Float32, EMBED_DIM))
    ).write_parquet(out)
    print(f"{len(rows)} vectors, norms {norms.min():.3f}-{norms.max():.3f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
