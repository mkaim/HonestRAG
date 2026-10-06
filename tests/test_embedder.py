import numpy as np
import pytest

from embedders import sentence_transformer
from embedders.sentence_transformer import SenTranEmbedder

MAX_SEQ_LENGTH = 32
DIMS = 3


@pytest.fixture
def encoded(monkeypatch, hf_tokenizer) -> list[str]:
    """Replaces SentenceTransformer with a fake around the real tokenizer, so
    no model weights are loaded. Returns the texts passed to encode()."""
    calls: list[str] = []

    class FakeSentenceTransformer:
        tokenizer = hf_tokenizer
        max_seq_length = MAX_SEQ_LENGTH

        def __init__(self, model: str):
            pass

        def get_embedding_dimension(self) -> int:
            return DIMS

        def encode(self, texts: list[str], **kwargs) -> np.ndarray:
            calls.extend(texts)
            return np.zeros((len(texts), DIMS))

    monkeypatch.setattr(
        sentence_transformer, "SentenceTransformer", FakeSentenceTransformer
    )
    return calls


def _embedder() -> SenTranEmbedder:
    return SenTranEmbedder("fake", query_prefix="query: ", document_prefix="passage: ")


def test_limits_exclude_special_tokens_and_document_prefix(encoded):
    embedder = _embedder()
    prefix_tokens = embedder.tokenizer.count("passage: ")

    assert embedder.dims == DIMS
    assert embedder.max_input_tokens == MAX_SEQ_LENGTH - 2
    assert embedder.tokenizer.max_tokens == embedder.max_input_tokens - prefix_tokens


async def test_queries_and_documents_get_their_prefix(encoded):
    embedder = _embedder()

    vectors = await embedder.embed_queries(["what is rag"])
    await embedder.embed_documents(["rag is retrieval"])

    assert encoded == ["query: what is rag", "passage: rag is retrieval"]
    assert vectors == [[0.0] * DIMS]


async def test_chunk_at_tokenizer_limit_fits_with_prefix(encoded):
    embedder = _embedder()
    text = embedder.tokenizer.split("word " * 100, embedder.tokenizer.max_tokens)[0]

    await embedder.embed_documents([text])

    assert encoded == [f"passage: {text}"]


async def test_embed_rejects_text_over_limit_before_encoding(encoded):
    embedder = _embedder()

    with pytest.raises(ValueError, match="token limit"):
        await embedder.embed_documents(["word " * 100])
    assert encoded == []


def test_dims_falls_back_to_legacy_method_name(encoded, monkeypatch):
    fake = sentence_transformer.SentenceTransformer
    monkeypatch.delattr(fake, "get_embedding_dimension")
    monkeypatch.setattr(
        fake, "get_sentence_embedding_dimension", lambda self: 7, raising=False
    )

    assert _embedder().dims == 7
