import pytest

from loaders import text_to_blocks
from models import Paragraph


def test_text_to_blocks_splits_on_blank_lines():
    blocks = text_to_blocks("first\n\nsecond\n\n\n\nthird", max_size=100)

    assert blocks == [Paragraph("first"), Paragraph("second"), Paragraph("third")]


def test_text_to_blocks_hard_splits_oversized_paragraphs():
    blocks = text_to_blocks("0123456789", max_size=4)

    assert [b.text for b in blocks] == ["0123", "4567", "89"]
    assert all(b.level is None for b in blocks)


def test_text_to_blocks_requires_positive_max_size():
    with pytest.raises(ValueError):
        text_to_blocks("x", max_size=0)
