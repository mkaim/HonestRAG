from pathlib import Path

from models import Block, Loader, Paragraph


def text_to_blocks(text: str, max_size: int) -> list[Paragraph]:
    """Split plain text on blank lines into paragraphs, hard-splitting any
    paragraph longer than max_size into smaller paragraphs."""

    paragraphs = [Paragraph(s.strip()) for s in text.split("\n\n") if s.strip()]
    return [
        Paragraph(p.text[i : i + max_size])
        for p in paragraphs
        for i in range(0, len(p.text), max_size)
    ]


class TextFileLoader(Loader):
    """Reads a plain text/markdown file, splitting it on blank lines into
    paragraphs no longer than max_size. `source` is a filesystem path."""

    def __init__(self, max_size: int):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size

    def load(self, source: str) -> tuple[list[Block], dict]:
        return text_to_blocks(Path(source).read_text(), self.max_size), {}
