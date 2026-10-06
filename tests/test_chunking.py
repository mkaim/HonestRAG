import pytest

from chunkers import BlockChunker
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


def _chunker(
    content_tokens: int, overlap_tokens: int = 0, breadcrumb_tokens: int = 64
) -> BlockChunker:
    return BlockChunker(
        CHAR_TOKENIZER,
        content_tokens=content_tokens,
        overlap_tokens=overlap_tokens,
        breadcrumb_tokens=breadcrumb_tokens,
    )


def _paragraphs(*texts: str) -> list[Paragraph]:
    return [Paragraph(t) for t in texts]


def test_chunk_packs_paragraphs_without_overlap():
    blocks = _paragraphs("aaa", "bbb", "ccc", "ddd")
    chunks = list(_chunker(8).chunk("doc1", blocks))

    assert [c.content for c in chunks] == ["aaa\n\nbbb", "ccc\n\nddd"]
    assert [c.embedding_text for c in chunks] == ["aaa\n\nbbb", "ccc\n\nddd"]


def test_chunk_fixed_size_overlap_within_same_level():
    blocks = _paragraphs("aaa", "bbb", "ccc", "ddd")
    chunks = list(_chunker(8, overlap_tokens=2).chunk("doc1", blocks))

    assert [c.content for c in chunks] == ["aaa\n\nbbb", "ccc\n\nddd"]
    assert [c.embedding_text for c in chunks] == [
        "aaa\n\nbbb",
        "bb\n\nccc\n\nddd",
    ]


def test_chunk_adds_header_breadcrumb_to_embedding_text():
    blocks = [Header("Title", level=1), Paragraph("body", level=1)]
    chunks = list(_chunker(100).chunk("doc1", blocks))

    assert chunks[0].content == "body"
    assert chunks[0].embedding_text == "[Title]\nbody"


def test_chunk_nested_header_breadcrumb():
    blocks = [
        Header("H1", level=1),
        Header("H1.1", level=2),
        Paragraph("deep", level=2),
    ]
    chunks = list(_chunker(100).chunk("doc1", blocks))

    assert chunks[0].embedding_text == "[H1 > H1.1]\ndeep"


def test_chunk_no_overlap_when_breadcrumb_changes():
    blocks = [
        Header("H1", level=1),
        Paragraph("A", level=1),
        Header("H1.1", level=2),
        Paragraph("C", level=2),
    ]
    chunks = list(_chunker(100, overlap_tokens=50).chunk("doc1", blocks))

    assert [c.content for c in chunks] == ["A", "C"]
    assert chunks[1].embedding_text == "[H1 > H1.1]\nC"


def test_chunk_shallower_paragraph_returns_to_parent_header():
    blocks = [
        Header("H1", level=1),
        Header("H1.1", level=2),
        Paragraph("deep", level=2),
        Paragraph("shallow", level=1),
    ]
    chunks = list(_chunker(100).chunk("doc1", blocks))

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
    chunks = list(_chunker(6, overlap_tokens=2).chunk("doc1", blocks))

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
    chunks = list(_chunker(6, overlap_tokens=2).chunk("doc1", blocks))

    assert [c.content for c in chunks] == ["P1\n\nP2", "P3"]
    assert [c.embedding_text for c in chunks] == [
        "[H1]\nP1\n\nP2",
        "[H1]\nP2\n\nP3",
    ]


def test_chunk_splits_oversized_paragraph():
    chunks = list(_chunker(10).chunk("doc1", _paragraphs("x" * 50)))

    assert [c.content for c in chunks] == ["x" * 10] * 5
    assert [c.embedding_text for c in chunks] == ["x" * 10] * 5


def test_chunk_splits_oversized_paragraph_with_overlap():
    chunks = list(_chunker(4, overlap_tokens=2).chunk("doc1", _paragraphs("abcdef")))

    assert [c.content for c in chunks] == ["abcd", "ef"]
    assert [c.embedding_text for c in chunks] == ["abcd", "cd\n\nef"]


def test_chunk_empty_blocks_yields_nothing():
    assert list(_chunker(400).chunk("doc1", [])) == []


def test_chunk_validation():
    with pytest.raises(ValueError):
        BlockChunker(CHAR_TOKENIZER, content_tokens=0)
    with pytest.raises(ValueError):
        BlockChunker(CHAR_TOKENIZER, content_tokens=10, overlap_tokens=-1)
    with pytest.raises(ValueError):
        BlockChunker(CHAR_TOKENIZER, content_tokens=10, breadcrumb_tokens=-1)
    with pytest.raises(ValueError, match="token limit is too small"):
        BlockChunker(CharTokenizer(max_tokens=10), content_tokens=10)


def test_chunk_same_level_header_replaces_breadcrumb():
    blocks = [
        Header("H1", level=1),
        Paragraph("A", level=1),
        Header("H2", level=1),
        Paragraph("B", level=1),
    ]
    chunks = list(_chunker(100).chunk("doc1", blocks))

    assert [c.content for c in chunks] == ["A", "B"]
    assert [c.embedding_text for c in chunks] == ["[H1]\nA", "[H2]\nB"]


def test_chunk_header_level_jump():
    blocks = [
        Header("H1", level=1),
        Header("H3", level=3),
        Paragraph("deep", level=3),
    ]
    chunks = list(_chunker(100).chunk("doc1", blocks))

    assert chunks[0].embedding_text == "[H1 > H3]\ndeep"


def test_chunk_deep_stack_pop():
    blocks = [
        Header("H1", level=1),
        Header("H1.1", level=2),
        Header("H1.1.1", level=3),
        Paragraph("deep", level=3),
        Paragraph("top", level=1),
    ]
    chunks = list(_chunker(100).chunk("doc1", blocks))

    assert [c.content for c in chunks] == ["deep", "top"]
    assert [c.embedding_text for c in chunks] == [
        "[H1 > H1.1 > H1.1.1]\ndeep",
        "[H1]\ntop",
    ]


def test_chunk_overlap_uses_full_previous_content_when_shorter():
    chunks = list(
        _chunker(4, overlap_tokens=10).chunk("doc1", _paragraphs("aa", "bb", "cc"))
    )

    assert [c.content for c in chunks] == ["aa", "bb", "cc"]
    assert [c.embedding_text for c in chunks] == ["aa", "aa\n\nbb", "bb\n\ncc"]


def test_chunk_skips_empty_paragraphs():
    blocks = [Paragraph("A"), Paragraph(""), Paragraph("B")]
    chunks = list(_chunker(100).chunk("doc1", blocks))

    assert [c.content for c in chunks] == ["A\n\nB"]
    assert chunks[0].embedding_text == "A\n\nB"


def test_chunk_respects_embedder_token_limit():
    small = CharTokenizer(max_tokens=40)
    with pytest.warns(UserWarning):
        chunker = BlockChunker(
            small, content_tokens=400, overlap_tokens=0, breadcrumb_tokens=20
        )

    # 40 max tokens - 0 overlap - 20 breadcrumb - 4 join
    assert chunker.content_tokens == 16


def _breadcrumb_of(names: list[str], breadcrumb_tokens: int) -> str:
    blocks = [Header(n, level=i) for i, n in enumerate(names, 1)]
    blocks.append(Paragraph("body"))
    chunker = _chunker(100, breadcrumb_tokens=breadcrumb_tokens)
    (chunk,) = chunker.chunk("doc1", blocks)
    return chunk.embedding_text.removesuffix("body")


def test_breadcrumb_full_path_when_it_fits():
    assert _breadcrumb_of(["R", "M1", "M2", "L"], 17) == "[R > M1 > M2 > L]\n"


def test_breadcrumb_drops_middle_levels_farthest_from_leaf_first():
    assert _breadcrumb_of(["R", "M1", "M2", "L"], 16) == "[R > … > M2 > L]\n"
    assert _breadcrumb_of(["R", "M1", "M2", "L"], 15) == "[R > … > L]\n"


def test_breadcrumb_clips_long_root_before_dropping_it():
    assert _breadcrumb_of(["Rootname", "L"], 12) == "[Rootn… > L]\n"


def test_breadcrumb_clips_root_and_marks_dropped_middle():
    assert _breadcrumb_of(["Rootname", "M", "L"], 15) == "[Root… > … > L]\n"


def test_breadcrumb_falls_back_to_leaf_only():
    assert _breadcrumb_of(["R", "L"], 5) == "[L]\n"


def test_breadcrumb_clips_long_leaf():
    assert _breadcrumb_of(["abcdefghijklmnop"], 10) == "[abcdefg…]\n"


def test_breadcrumb_omitted_when_budget_too_small():
    assert _breadcrumb_of(["Title"], 2) == ""


def test_embedding_text_never_exceeds_embedder_limit():
    tokenizer = CharTokenizer(max_tokens=60)
    with pytest.warns(UserWarning):
        chunker = BlockChunker(
            tokenizer, content_tokens=400, overlap_tokens=8, breadcrumb_tokens=20
        )
    blocks = [
        Header("A very long top level title " * 3, level=1),
        Header("Middle", level=2),
        Header("Another long section heading " * 2, level=3),
        Paragraph("word " * 40, level=3),
        Paragraph("short", level=3),
        Paragraph("tail " * 30, level=1),
    ]

    chunks = list(chunker.chunk("doc1", blocks))

    assert len(chunks) > 3
    assert all(len(c.embedding_text) <= tokenizer.max_tokens for c in chunks)
