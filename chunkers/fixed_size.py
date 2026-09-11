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

        index = 0
        for start in range(0, max(len(text), 1), step):
            piece = text[start : start + self.size]
            if not piece:
                break
            yield Chunk(document_id=document_id, index=index, content=piece)
            index += 1
            if start + self.size >= len(text):
                break
