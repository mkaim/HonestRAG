from pathlib import Path

from models import Block, Loader, Paragraph


def text_to_blocks(text: str) -> list[Paragraph]:
    """Split plain text on blank lines; each non-empty span is a Paragraph."""
    return [Paragraph(s.strip()) for s in text.split("\n\n") if s.strip()]


class TextFileLoader(Loader):
    """Reads a plain text/markdown file, splitting it on blank lines into
    paragraphs. `source` is a filesystem path."""

    def load(self, source: str) -> tuple[list[Block], dict]:
        return text_to_blocks(Path(source).read_text()), {}
