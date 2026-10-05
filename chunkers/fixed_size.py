from collections.abc import Iterator

from models import Chunk, Chunker, DocumentID


class FixedSizeChunker(Chunker):
    """Splits text into fixed-size, overlapping character windows. Simple
    and model-agnostic; not token-aware."""

    def __init__(self, size: int = 1000, overlap: int = 100):
        if overlap * 2 > size:
            raise ValueError("overlap cannot exceed half of size")
        self.size = size
        self.overlap = overlap

    def chunk(self, document_id: DocumentID, text: str) -> Iterator[Chunk]:
        step = self.size - self.overlap

        for index, start in enumerate(range(0, len(text), step)):
            piece = text[start : start + self.size]
            yield Chunk(document_id=document_id, index=index, content=piece)
            # Stop once the window reaches the end, avoiding a redundant tail.
            if start + self.size >= len(text):
                break
