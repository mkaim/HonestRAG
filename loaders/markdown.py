import re
from pathlib import Path

from models import Block, Header, Loader, Paragraph

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def markdown_to_blocks(text: str) -> list[Block]:
    """Split markdown on blank lines; each span is a Header if it starts with
    `#`..`######`, otherwise a Paragraph. No other markdown syntax is
    interpreted."""
    blocks: list[Block] = []
    for span in text.split("\n\n"):
        span = span.strip()
        if not span:
            continue
        match = _HEADER_RE.match(span)
        if match:
            blocks.append(Header(match.group(2).strip(), level=len(match.group(1))))
        else:
            blocks.append(Paragraph(span))
    return blocks


class MarkdownFileLoader(Loader):
    """Reads a markdown file, splitting it on blank lines into headers and
    paragraphs. `source` is a filesystem path."""

    def load(self, source: str) -> tuple[list[Block], dict]:
        return markdown_to_blocks(Path(source).read_text()), {}
