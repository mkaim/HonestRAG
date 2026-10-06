import warnings
from collections.abc import Callable, Iterator

from models import Block, Chunk, Chunker, DocumentID, Header, Tokenizer

Breadcrumb = tuple[Header, ...]
_PARAGRAPH_SEP = "\n\n"
_ELLIPSIS = "…"
# Room for the separators joining breadcrumb, overlap and content.
_JOIN_TOKENS = 4


def _render(parts: list[str]) -> str:
    return f"[{' > '.join(parts)}]" if parts else ""


class BlockChunker(Chunker):
    """Token-aware, header-aware chunker. Packs consecutive paragraphs into
    chunks whose stored content is at most `content_tokens` tokens, splitting
    paragraphs longer than that into multiple chunks. Each chunk also carries
    an embedding_text: the stored content plus a [HEADER > ...] breadcrumb
    prefix of at most `breadcrumb_tokens` tokens and, when continuing the same
    section, up to `overlap_tokens` trailing tokens of the previous chunk's
    content."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        content_tokens: int,
        overlap_tokens: int = 40,
        breadcrumb_tokens: int = 64,
    ):
        if content_tokens <= 0 or overlap_tokens < 0 or breadcrumb_tokens < 0:
            raise ValueError(
                "content_tokens must be positive; overlap_tokens and "
                "breadcrumb_tokens non-negative"
            )
        self.tokenizer = tokenizer
        self.overlap_tokens = overlap_tokens
        self.breadcrumb_tokens = breadcrumb_tokens
        self.content_tokens = min(
            content_tokens,
            tokenizer.max_tokens - overlap_tokens - breadcrumb_tokens - _JOIN_TOKENS,
        )
        if self.content_tokens <= 0:
            raise ValueError(
                "embedder token limit is too small for overlap and breadcrumb"
            )
        if self.content_tokens < content_tokens:
            warnings.warn(
                f"embedder max_tokens={tokenizer.max_tokens} limits chunks to "
                f"{self.content_tokens} content tokens "
                f"(requested {content_tokens})",
                stacklevel=2,
            )

    def chunk(self, document_id: DocumentID, blocks: list[Block]) -> Iterator[Chunk]:
        prev_breadcrumb: Breadcrumb | None = None
        overlap_text = ""
        prefix = ""
        for index, (breadcrumb, content) in enumerate(self._groups(blocks)):
            if prev_breadcrumb != breadcrumb:
                overlap_text = ""
                prefix = self._breadcrumb(breadcrumb)

            body = (
                f"{overlap_text}{_PARAGRAPH_SEP}{content}" if overlap_text else content
            )
            yield Chunk(
                document_id=document_id,
                index=index,
                content=content,
                embedding_text=f"{prefix}\n{body}" if prefix else body,
            )
            prev_breadcrumb = breadcrumb
            overlap_text = self.tokenizer.tail(content, self.overlap_tokens)

    def _breadcrumb(self, headers: Breadcrumb) -> str:
        """Render headers as a [A > B > C] prefix of at most
        `breadcrumb_tokens` tokens. When the full path doesn't fit, keeps the
        leaf (most specific), then the root (document context), then middle
        levels nearest the leaf, marking dropped levels with "…". A header
        that can't fit whole is cut at a token boundary."""
        names = [h.text for h in headers]
        if self._fits(names):
            return _render(names)

        leaf = self._clip(names[-1], lambda s: [s])
        if not leaf:
            return ""
        if len(names) == 1:
            return _render([leaf])

        middle = names[1:-1]
        gap = [_ELLIPSIS] if middle else []
        root = self._clip(names[0], lambda s: [s, *gap, leaf])
        if not root:
            return _render([leaf])

        # The full path didn't fit, so keep as many middle levels nearest the
        # leaf as fit, down to none.
        for keep in range(len(middle) - 1, 0, -1):
            parts = [root, _ELLIPSIS, *middle[-keep:], leaf]
            if self._fits(parts):
                return _render(parts)
        return _render([root, *gap, leaf])

    def _fits(self, parts: list[str]) -> bool:
        return self.tokenizer.count(_render(parts)) <= self.breadcrumb_tokens

    def _clip(self, text: str, layout: Callable[[str], list[str]]) -> str:
        """Longest token prefix of `text` (suffixed with "…" when cut) whose
        `layout` fits the breadcrumb budget, or "" if none does."""
        if self._fits(layout(text)):
            return text
        best = ""
        lo, hi = 1, self.tokenizer.count(text) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            clipped = self.tokenizer.split(text, mid)[0] + _ELLIPSIS
            if self._fits(layout(clipped)):
                best, lo = clipped, mid + 1
            else:
                hi = mid - 1
        return best

    def _groups(self, blocks: list[Block]) -> Iterator[tuple[Breadcrumb, str]]:
        """Group paragraph pieces into joined chunk content, breaking on
        headers, breadcrumb changes, and when the token count would exceed
        `content_tokens`."""
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
            if content and new_breadcrumb != breadcrumb:
                yield breadcrumb, content
                content = ""
            breadcrumb = new_breadcrumb

            for piece in self.tokenizer.split(block.text, self.content_tokens):
                candidate = f"{content}{_PARAGRAPH_SEP}{piece}" if content else piece
                if content and self.tokenizer.count(candidate) > self.content_tokens:
                    yield breadcrumb, content
                    candidate = piece
                content = candidate

        if content:
            yield breadcrumb, content
