from collections.abc import Iterator

from models import Block, Chunk, Chunker, DocumentID, Header

Breadcrumb = tuple[Header, ...]
_PARAGRAPH_SEP = "\n\n"


def _breadcrumb(headers: Breadcrumb) -> str:
    if not headers:
        return ""
    return f"[{' > '.join(h.text for h in headers)}]"


class BlockChunker(Chunker):
    """Header-aware chunker. Packs consecutive paragraphs into chunks whose
    stored content is at most `size` characters, splitting only at paragraph
    boundaries. Each chunk also carries an embedding_text: the stored content
    plus a [HEADER > ...] breadcrumb prefix and, when continuing the same
    section, up to `overlap` trailing characters of the previous chunk's
    content."""

    def __init__(self, size: int = 1000, overlap: int = 100):
        if size <= 0 or overlap < 0:
            raise ValueError("size must be positive and overlap must be non-negative")
        self.size = size
        self.overlap = overlap

    def chunk(self, document_id: DocumentID, blocks: list[Block]) -> Iterator[Chunk]:
        prev_breadcrumb: Breadcrumb | None = None
        overlap_text = ""
        for index, (breadcrumb, content) in enumerate(self._groups(blocks)):
            if prev_breadcrumb != breadcrumb:
                overlap_text = ""

            body = (
                f"{overlap_text}{_PARAGRAPH_SEP}{content}"
                if overlap_text
                else content
            )
            prefix = _breadcrumb(breadcrumb)
            yield Chunk(
                document_id=document_id,
                index=index,
                content=content,
                embedding_text=f"{prefix}\n{body}" if prefix else body,
            )
            prev_breadcrumb = breadcrumb
            overlap_text = content[-self.overlap :] if self.overlap else ""

    def _groups(self, blocks: list[Block]) -> Iterator[tuple[Breadcrumb, str]]:
        """Group paragraph texts into joined chunk content, breaking on
        headers, breadcrumb changes, and when the content would exceed
        `size`."""
        headers: list[Header] = []
        breadcrumb: Breadcrumb = ()
        content = ""

        for block in blocks:
            if isinstance(block, Header):
                if content:
                    yield breadcrumb, content
                    content = ""
                headers = headers[: block.level - 1] + [block]
                continue
            if not block.text:
                continue

            if block.level is not None:
                headers = headers[: block.level]
            new_breadcrumb = tuple(headers)
            if content and (
                new_breadcrumb != breadcrumb
                or len(content) + len(_PARAGRAPH_SEP) + len(block.text) > self.size
            ):
                yield breadcrumb, content
                content = ""

            content = (
                f"{content}{_PARAGRAPH_SEP}{block.text}" if content else block.text
            )
            breadcrumb = new_breadcrumb

        if content:
            yield breadcrumb, content
