from collections.abc import Iterator

from models import Block, Chunk, DocumentID, Tokenizer

_PARAGRAPH_SEP = "\n\n"


class FixedSizeChunker:
    """Naive baseline: ignores block structure entirely (headers and
    paragraphs are flattened into one text, in order) and slides a
    fixed-size token window across it, each chunk preceded by the last
    `overlap_tokens` tokens of the previous one. No breadcrumb, no
    header-aware packing - embedding_text always equals content. Exists to
    give BlockChunker's token-aware, header-aware packing something to beat
    in eval/, not as a chunker meant for production ingestion."""

    def __init__(self, tokenizer: Tokenizer, content_tokens: int, overlap_tokens: int = 0):
        if content_tokens <= 0 or overlap_tokens < 0:
            raise ValueError(
                "content_tokens must be positive; overlap_tokens non-negative"
            )
        self.tokenizer = tokenizer
        self.content_tokens = content_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, document_id: DocumentID, blocks: list[Block]) -> Iterator[Chunk]:
        remaining = _PARAGRAPH_SEP.join(b.text for b in blocks if b.text)

        index = 0
        overlap = ""
        while remaining:
            content = self.tokenizer.split(remaining, self.content_tokens)[0]
            remaining = remaining[len(content) :].lstrip()

            embedding_text = (
                f"{overlap}{_PARAGRAPH_SEP}{content}" if overlap else content
            )
            yield Chunk(
                document_id=document_id,
                index=index,
                content=content,
                embedding_text=embedding_text,
            )

            overlap = self.tokenizer.tail(content, self.overlap_tokens)
            index += 1
