import numpy as np
import pytest

torch = pytest.importorskip("torch")

from locus.embed_specter2 import EMBED_DIM, check, embed_texts


class _Batch(dict):
    def to(self, device):
        return self


class _Tokenizer:
    def __call__(self, batch, **kwargs):
        return _Batch(texts=batch)


class _Output:
    def __init__(self, tensor):
        self.last_hidden_state = tensor


class _Model:
    """Encodes each text's length, so a vector identifies the text it came from."""

    def __call__(self, texts):
        rows = [[float(len(t))] * EMBED_DIM for t in texts]
        return _Output(torch.tensor(rows).unsqueeze(1))


def test_length_sort_does_not_leak_into_output_order():
    # embed_texts batches longest-first for throughput. The output is keyed
    # positionally against the input frame, so if that sort survived into the
    # returned array every context would be scored against another context's
    # vector -- a silent, total corruption that still produces plausible MRRs.
    texts = ["a" * n for n in (3, 40, 1, 25, 7, 60, 12, 2, 33)]
    out = embed_texts(texts, _Tokenizer(), _Model(), "cpu", batch_size=2)
    assert out.shape == (len(texts), EMBED_DIM)
    assert [int(v[0]) for v in out] == [len(t) for t in texts]


def test_single_batch_is_also_restored():
    texts = ["a" * n for n in (5, 1, 9)]
    out = embed_texts(texts, _Tokenizer(), _Model(), "cpu", batch_size=64)
    assert [int(v[0]) for v in out] == [5, 1, 9]


def test_check_refuses_nan():
    v = np.ones((3, EMBED_DIM), dtype=np.float32)
    v[1, 0] = np.nan
    with pytest.raises(SystemExit, match="NaN"):
        check(v, "test")


def test_check_refuses_zero_vectors():
    # A zero vector has undefined cosine; numpy yields nan or 0 depending on
    # the expression, and either way that candidate is not really scored.
    v = np.ones((3, EMBED_DIM), dtype=np.float32)
    v[2] = 0.0
    with pytest.raises(SystemExit, match="zero vectors"):
        check(v, "test")


def test_check_reports_norms_on_a_healthy_matrix():
    v = np.ones((4, EMBED_DIM), dtype=np.float32)
    stats = check(v, "test")
    assert stats["rows"] == 4
    assert stats["norm_min"] == pytest.approx(np.sqrt(EMBED_DIM))
