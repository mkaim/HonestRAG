from pathlib import Path

from loaders import DocxFileLoader, PdfFileLoader, markdown_to_blocks, text_to_blocks
from models import Header, Paragraph

_FIXTURES = Path(__file__).parent / "fixtures"


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


def test_docx_loader_converts_headings_and_paragraphs():
    blocks, metadata = DocxFileLoader().load(str(_FIXTURES / "sample.docx"))

    assert blocks[0] == Header("Chapter One", level=1)
    assert blocks[1] == Paragraph("Intro paragraph under chapter one.")
    assert blocks[2] == Header("A Subsection", level=2)
    assert blocks[3] == Paragraph("Body text in the subsection.")
    assert metadata == {}


def test_docx_loader_converts_table_to_markdown_pipes():
    blocks, _ = DocxFileLoader().load(str(_FIXTURES / "sample.docx"))

    table_block = blocks[-1]
    assert isinstance(table_block, Paragraph)
    assert "ITEM" in table_block.text
    assert "Widget" in table_block.text
    assert "|" in table_block.text


def test_pdf_loader_detects_bold_larger_text_as_header():
    blocks, metadata = PdfFileLoader().load(str(_FIXTURES / "sample.pdf"))

    assert blocks[0] == Header("Document Title", level=1)
    assert metadata == {}


def test_pdf_loader_keeps_body_text_as_paragraphs():
    blocks, _ = PdfFileLoader().load(str(_FIXTURES / "sample.pdf"))

    paragraphs = [b for b in blocks if isinstance(b, Paragraph)]
    assert paragraphs == [
        Paragraph("This is the first body paragraph of the document."),
        Paragraph("This is a second body paragraph with more text in it."),
    ]
