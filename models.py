from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

DocumentID = str
ChunkID = str


class SearchMode(StrEnum):
    BM25 = "BM25"
    SEMANTIC = "SEMANTIC"


@dataclass
class Document:
    """Parent record for a source: groups its Chunks and holds source-level
    metadata (author, title, ...) once instead of on every Chunk. Holds no
    content itself - content is loaded separately and only ever stored as
    Chunks. `id` is the caller's own identifier for the source (a file path,
    a URL, ...)."""

    id: DocumentID
    metadata: dict[str, str | int | float] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrieval unit: one piece of a Document's text, as produced by a
    Chunker. `id` is derived from document_id + index, never set directly, so
    it can never drift out of sync with them. `metadata` is this chunk's own
    (e.g. page number, offsets) - not a copy of the parent Document's."""

    document_id: DocumentID
    index: int
    content: str
    metadata: dict[str, str | int | float] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.document_id}#{self.index}"


@dataclass(frozen=True)
class SearchResult:
    """Immutable. Build one as SearchResult(chunk=c).with_score(name, value),
    chaining with_score() for each score a stage (retrieval, fusion, reranking)
    contributes. There is no other way to set a score, so a SearchResult can be
    freely shared/reused across stages without risk of one stage's write
    leaking into another's."""

    chunk: Chunk
    scores: MappingProxyType[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def with_score(self, name: str, value: float) -> SearchResult:
        return SearchResult(self.chunk, MappingProxyType({**self.scores, name: value}))

    def score(self, name: str) -> float:
        return self.scores[name]


class Loader(Protocol):
    def load(self, source: str) -> tuple[str, dict[str, str | int | float]]:
        """Extract text and any metadata discoverable from the source itself
        (e.g. a Notion page's title). Metadata is empty if the format has
        nothing to offer (e.g. plain text). The caller decides how to merge
        this with metadata it already knows when building a Document."""
        ...


class Chunker(Protocol):
    def chunk(self, document_id: DocumentID, text: str) -> Iterator[Chunk]: ...


class Embedder(Protocol):
    model: str
    dims: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class RankFusion(Protocol):
    def merge(
        self, rankings: list[list[SearchResult]], limit: int
    ) -> list[SearchResult]: ...


class CrossEncoder(Protocol):
    async def score(
        self, query: str, results: list[SearchResult], limit: int
    ) -> list[SearchResult]: ...


class RagDb(Protocol):
    async def search(
        self, mode: SearchMode, queries: list[str], limit: int
    ) -> list[list[SearchResult]]: ...

    async def add(self, document: Document, chunks: list[Chunk]) -> None: ...


class RagBase(Protocol):
    async def search(self, queries: list[str]) -> list[list[SearchResult]]: ...
