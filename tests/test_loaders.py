from loaders import markdown_to_blocks, text_to_blocks
from models import Header, Paragraph


def test_text_to_blocks_splits_on_blank_lines():
    blocks = text_to_blocks("first\n\nsecond\n\n\n\nthird")

    assert blocks == [Paragraph("first"), Paragraph("second"), Paragraph("third")]


def test_text_to_blocks_keeps_long_paragraph_whole():
    blocks = text_to_blocks("0123456789")

    assert blocks == [Paragraph("0123456789")]
    assert blocks[0].level is None


def test_markdown_to_blocks_detects_headers():
    blocks = markdown_to_blocks("# Title\n\nsome text\n\n## Section\n\nmore text")

    assert blocks == [
        Header("Title", level=1),
        Paragraph("some text"),
        Header("Section", level=2),
        Paragraph("more text"),
    ]


def test_markdown_to_blocks_skips_blank_spans():
    blocks = markdown_to_blocks("first\n\n\n\nsecond")

    assert blocks == [Paragraph("first"), Paragraph("second")]
