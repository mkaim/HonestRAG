import pytest

from chunkers import BlockChunker
from models import Header, Paragraph


def _paragraphs(*texts: str) -> list[Paragraph]:
    return [Paragraph(t) for t in texts]


def test_chunk_packs_paragraphs_without_overlap():
    blocks = _paragraphs("aaa", "bbb", "ccc", "ddd")
    chunks = list(BlockChunker(size=8, overlap=0).chunk("doc1", blocks))

    assert [c.content for c in chunks] == ["aaa\n\nbbb", "ccc\n\nddd"]
    assert [c.embedding_text for c in chunks] == ["aaa\n\nbbb", "ccc\n\nddd"]


def test_chunk_fixed_size_overlap_within_same_level():
    blocks = _paragraphs("aaa", "bbb", "ccc", "ddd")
    chunks = list(BlockChunker(size=8, overlap=2).chunk("doc1", blocks))

    assert [c.content for c in chunks] == ["aaa\n\nbbb", "ccc\n\nddd"]
    assert [c.embedding_text for c in chunks] == [
        "aaa\n\nbbb",
        "bb\n\nccc\n\nddd",
    ]


def test_chunk_adds_header_breadcrumb_to_embedding_text():
    blocks = [Header("Title", level=1), Paragraph("body", level=1)]
    chunks = list(BlockChunker(size=100, overlap=0).chunk("doc1", blocks))

    assert chunks[0].content == "body"
    assert chunks[0].embedding_text == "[Title]\nbody"


def test_chunk_nested_header_breadcrumb():
    blocks = [
        Header("H1", level=1),
        Header("H1.1", level=2),
        Paragraph("deep", level=2),
    ]
    chunks = list(BlockChunker(size=100, overlap=0).chunk("doc1", blocks))

    assert chunks[0].embedding_text == "[H1 > H1.1]\ndeep"


def test_chunk_no_overlap_when_breadcrumb_changes():
    blocks = [
        Header("H1", level=1),
        Paragraph("A", level=1),
        Header("H1.1", level=2),
        Paragraph("C", level=2),
    ]
    chunks = list(BlockChunker(size=100, overlap=50).chunk("doc1", blocks))

    assert [c.content for c in chunks] == ["A", "C"]
    assert chunks[1].embedding_text == "[H1 > H1.1]\nC"


def test_chunk_shallower_paragraph_returns_to_parent_header():
    blocks = [
        Header("H1", level=1),
        Header("H1.1", level=2),
        Paragraph("deep", level=2),
        Paragraph("shallow", level=1),
    ]
    chunks = list(BlockChunker(size=100, overlap=0).chunk("doc1", blocks))

    assert [c.content for c in chunks] == ["deep", "shallow"]
    assert [c.embedding_text for c in chunks] == [
        "[H1 > H1.1]\ndeep",
        "[H1]\nshallow",
    ]


def test_chunk_overlaps_fixed_tail_within_same_section():
    blocks = [
        Header("H1", level=1),
        Paragraph("P1", level=1),
        Paragraph("P2", level=1),
        Paragraph("P3", level=1),
    ]
    chunks = list(BlockChunker(size=6, overlap=2).chunk("doc1", blocks))

    assert [c.content for c in chunks] == ["P1\n\nP2", "P3"]
    assert [c.embedding_text for c in chunks] == [
        "[H1]\nP1\n\nP2",
        "[H1]\nP2\n\nP3",
    ]


def test_chunk_paragraph_without_level_inherits_current_header():
    blocks = [
        Header("H1", level=1),
        Paragraph("P1"),
        Paragraph("P2"),
        Paragraph("P3"),
    ]
    chunks = list(BlockChunker(size=6, overlap=2).chunk("doc1", blocks))

    assert [c.content for c in chunks] == ["P1\n\nP2", "P3"]
    assert [c.embedding_text for c in chunks] == [
        "[H1]\nP1\n\nP2",
        "[H1]\nP2\n\nP3",
    ]


def test_chunk_single_oversized_block_is_emitted_alone():
    chunks = list(
        BlockChunker(size=10, overlap=0).chunk("doc1", _paragraphs("x" * 50))
    )

    assert len(chunks) == 1
    assert chunks[0].content == "x" * 50
    assert chunks[0].embedding_text == "x" * 50


def test_chunk_empty_blocks_yields_nothing():
    assert list(BlockChunker().chunk("doc1", [])) == []


def test_size_and_overlap_validation():
    with pytest.raises(ValueError):
        BlockChunker(size=0)
    with pytest.raises(ValueError):
        BlockChunker(overlap=-1)
