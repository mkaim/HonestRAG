import pytest

from chunkers import BlockChunker
from embedders.sentence_transformer import HfTokenizer
from models import Header, Paragraph

TEXT = "Zażółć gęślą jaźń. The quick brown fox jumps over the lazy dog. " * 5


@pytest.fixture
def tokenizer(hf_tokenizer) -> HfTokenizer:
    return HfTokenizer(hf_tokenizer, max_tokens=96)


def test_count_excludes_special_tokens(tokenizer):
    assert tokenizer.count("") == 0
    assert tokenizer.count("hello") > 0


def test_split_returns_ordered_original_slices_within_limit(tokenizer):
    pieces = tokenizer.split(TEXT, 7)

    assert len(pieces) > 1
    assert all(tokenizer.count(p) <= 7 for p in pieces)
    position = 0
    for piece in pieces:
        position = TEXT.index(piece, position) + len(piece)
    assert "".join(pieces).replace(" ", "") == TEXT.replace(" ", "")


def test_tail_returns_suffix_within_limit(tokenizer):
    tail = tokenizer.tail(TEXT, 5)

    assert TEXT.endswith(tail)
    assert 0 < tokenizer.count(tail) <= 5
    assert tokenizer.tail(TEXT, 0) == ""
    assert tokenizer.tail("short", 100) == "short"


def test_chunker_embedding_text_fits_real_tokenizer(tokenizer):
    with pytest.warns(UserWarning):
        chunker = BlockChunker(
            tokenizer, content_tokens=400, overlap_tokens=16, breadcrumb_tokens=24
        )
    blocks = [
        Header("Introduction to the very long and verbose title " * 3, level=1),
        Header("Middle", level=2),
        Header("Zażółć gęślą jaźń section " * 4, level=3),
        Paragraph(TEXT, level=3),
        Paragraph(TEXT, level=1),
    ]

    chunks = list(chunker.chunk("doc1", blocks))

    assert len(chunks) > 2
    assert all(
        tokenizer.count(c.embedding_text) <= tokenizer.max_tokens for c in chunks
    )
    assert all(c.embedding_text.startswith("[") for c in chunks)
