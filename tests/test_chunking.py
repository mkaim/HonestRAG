import pytest

from chunkers import FixedSizeChunker


def test_chunk_covers_whole_document_with_overlap():
    text = "0123456789" * 25  # 250 chars
    chunker = FixedSizeChunker(size=100, overlap=20)

    chunks = list(chunker.chunk("doc1", text))

    assert [c.document_id for c in chunks] == ["doc1"] * len(chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert [c.id for c in chunks] == [f"doc1#{i}" for i in range(len(chunks))]

    # every position in the original text is covered by at least one chunk
    reconstructed = chunks[0].content
    for c in chunks[1:]:
        reconstructed += c.content[chunker.overlap :]
    assert reconstructed == text


def test_chunk_does_not_emit_redundant_tail():
    text = "0123456789" * 25  # 250 chars
    chunks = list(FixedSizeChunker(size=100, overlap=20).chunk("doc1", text))

    assert len(chunks) == 3
    assert chunks[-1].content == text[160:]


def test_chunk_short_document_is_a_single_chunk():
    chunks = list(FixedSizeChunker(size=1000, overlap=100).chunk("doc1", "short text"))

    assert len(chunks) == 1
    assert chunks[0].content == "short text"
    assert chunks[0].id == "doc1#0"


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        FixedSizeChunker(size=100, overlap=100)
