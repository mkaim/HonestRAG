from loaders import text_to_blocks
from models import Paragraph


def test_text_to_blocks_splits_on_blank_lines():
    blocks = text_to_blocks("first\n\nsecond\n\n\n\nthird")

    assert blocks == [Paragraph("first"), Paragraph("second"), Paragraph("third")]


def test_text_to_blocks_keeps_long_paragraph_whole():
    blocks = text_to_blocks("0123456789")

    assert blocks == [Paragraph("0123456789")]
    assert blocks[0].level is None
