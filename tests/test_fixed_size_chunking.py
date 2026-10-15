import pytest

from chunkers import FixedSizeChunker
from models import Header, Paragraph


class CharTokenizer:
    """1 character == 1 token; keeps chunker unit tests model-free."""

    def __init__(self, max_tokens: int = 100_000):
        self.max_tokens = max_tokens

    def count(self, text: str) -> int:
        return len(text)

    def split(self, text: str, limit: int) -> list[str]:
        return [text[i : i + limit] for i in range(0, len(text), limit)]

    def tail(self, text: str, n: int) -> str:
        return text[-n:] if n > 0 else ""


CHAR_TOKENIZER = CharTokenizer()


def _chunker(content_tokens: int, overlap_tokens: int = 0) -> FixedSizeChunker:
    return FixedSizeChunker(
        CHAR_TOKENIZER, content_tokens=content_tokens, overlap_tokens=overlap_tokens
    )


def test_splits_flattened_text_by_fixed_size():
    blocks = [Header("Title", level=1), Paragraph("0123456789")]
    chunks = list(_chunker(8).chunk("doc1", blocks))

    # headers and paragraphs are flattened together, ignoring block type:
    # "Title\n\n0123456789" (18 chars) cut every 8
    assert [c.content for c in chunks] == ["Title\n\n0", "12345678", "9"]
    assert [c.embedding_text for c in chunks] == ["Title\n\n0", "12345678", "9"]


def test_ignores_block_boundaries_mid_window():
    blocks = [Paragraph("aaa"), Paragraph("bbb"), Paragraph("ccc")]
    chunks = list(_chunker(6).chunk("doc1", blocks))

    # "aaa\n\nbbb\n\nccc" (14 chars) cut every 6, boundaries don't align
    assert [c.content for c in chunks] == ["aaa\n\nb", "bb\n\ncc", "c"]


def test_overlap_prepended_to_embedding_text_not_content():
    blocks = [Paragraph("0123456789abcdef")]
    chunks = list(_chunker(8, overlap_tokens=2).chunk("doc1", blocks))

    assert [c.content for c in chunks] == ["01234567", "89abcdef"]
    assert chunks[0].embedding_text == "01234567"  # no previous overlap yet
    assert chunks[1].embedding_text == "67\n\n89abcdef"


def test_empty_blocks_yield_nothing():
    assert list(_chunker(8).chunk("doc1", [])) == []
    assert list(_chunker(8).chunk("doc1", [Paragraph("")])) == []


def test_rejects_non_positive_content_tokens():
    with pytest.raises(ValueError):
        FixedSizeChunker(CHAR_TOKENIZER, content_tokens=0)


def test_rejects_negative_overlap_tokens():
    with pytest.raises(ValueError):
        FixedSizeChunker(CHAR_TOKENIZER, content_tokens=8, overlap_tokens=-1)
