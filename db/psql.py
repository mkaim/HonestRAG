import asyncio
import re

from pgvector import Vector
from pgvector.psycopg import register_vector_async
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from models import Chunk, Document, Embedder, RagDb, SearchMode, SearchResult

_BM25_SQL = """
SELECT doc.external_id, c.chunk_index, c.text, c.metadata,
       paradedb.score(c.id) AS score
FROM chunk c
JOIN document doc ON doc.id = c.document_id
WHERE c.id @@@ paradedb.match('text', %(query)s)
ORDER BY score DESC
LIMIT %(limit)s
"""


_SEMANTIC_SQL = """
SELECT doc.external_id, c.chunk_index, c.text, c.metadata,
        -(e.embedding <#> %(embedding)s) AS score
FROM {embed_table} e
JOIN chunk c ON c.id = e.chunk_id
JOIN document doc ON doc.id = c.document_id
ORDER BY e.embedding <#> %(embedding)s
LIMIT %(limit)s
"""

# Placeholders ({table}, {dims}, {embed_table}) are derived from the embedder
# model name/dims, never user input, so .format() interpolation is safe.
_DOCUMENT_UPSERT_SQL = """
INSERT INTO document (external_id, metadata)
VALUES (%s, %s)
ON CONFLICT (external_id)
DO UPDATE SET metadata = EXCLUDED.metadata
RETURNING id
"""

_CHUNK_UPSERT_SQL = """
INSERT INTO chunk (external_id, document_id, chunk_index, text, metadata)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (external_id)
DO UPDATE SET text = EXCLUDED.text, metadata = EXCLUDED.metadata
RETURNING id
"""

_CREATE_VECTOR_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    chunk_id BIGINT PRIMARY KEY
        REFERENCES chunk(id) ON DELETE CASCADE,
    embedding vector({dims}) NOT NULL
)
"""

_CREATE_VECTOR_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS {table}_hnsw_idx
    ON {table} USING hnsw (embedding vector_ip_ops)
"""

_EMBEDDING_UPSERT_SQL = """
INSERT INTO {table} (chunk_id, embedding)
VALUES (%s, %s)
ON CONFLICT (chunk_id)
DO UPDATE SET embedding = EXCLUDED.embedding
"""


class PsqlRagDb(RagDb):
    def __init__(self, conninfo: str, embedder: Embedder):
        self.conninfo = conninfo
        self.embedder = embedder
        self.pool = AsyncConnectionPool(
            conninfo, min_size=2, open=False, configure=register_vector_async
        )

    async def __aenter__(self) -> PsqlRagDb:
        await self.pool.open()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.pool.close()

    async def ensure_vector_table(self) -> str:
        """Create the per-model embedding table + HNSW index if missing."""
        table = self._vector_table_name()
        async with self.pool.connection() as conn:
            await conn.execute(
                _CREATE_VECTOR_TABLE_SQL.format(table=table, dims=self.embedder.dims)
            )
            await conn.execute(_CREATE_VECTOR_INDEX_SQL.format(table=table))
        return table

    async def add(self, document: Document, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        embeddings = await self.embedder.embed_documents(
            [c.embedding_text or c.content for c in chunks]
        )
        table = await self.ensure_vector_table()
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                _DOCUMENT_UPSERT_SQL,
                (document.id, Jsonb(document.metadata)),
            )
            (document_row_id,) = await cur.fetchone()

            for chunk, embedding in zip(chunks, embeddings, strict=True):
                cur = await conn.execute(
                    _CHUNK_UPSERT_SQL,
                    (
                        chunk.id,
                        document_row_id,
                        chunk.index,
                        chunk.content,
                        Jsonb(chunk.metadata),
                    ),
                )
                (chunk_row_id,) = await cur.fetchone()
                await conn.execute(
                    _EMBEDDING_UPSERT_SQL.format(table=table),
                    (chunk_row_id, Vector(embedding)),
                )

    async def search(
        self, mode: SearchMode, queries: list[str], limit: int
    ) -> list[list[SearchResult]]:
        match mode:
            case SearchMode.BM25:
                return await asyncio.gather(
                    *(self._search_bm25(q, limit) for q in queries)
                )
            case SearchMode.SEMANTIC:
                embeddings = await self.embedder.embed_queries(queries)
                return await asyncio.gather(
                    *(self._search_semantic(e, limit) for e in embeddings)
                )
            case _:
                raise ValueError(f"Unknown search mode: {mode}")

    async def _search_bm25(self, query: str, limit: int) -> list[SearchResult]:
        rows = await self._fetch(_BM25_SQL, {"query": query, "limit": limit})
        return self._to_results(rows, SearchMode.BM25)

    async def _search_semantic(
        self, embedding: list[float], limit: int
    ) -> list[SearchResult]:
        sql = _SEMANTIC_SQL.format(embed_table=self._vector_table_name())
        rows = await self._fetch(sql, {"embedding": Vector(embedding), "limit": limit})
        return self._to_results(rows, SearchMode.SEMANTIC)

    async def _fetch(self, sql: str, params: dict) -> list[tuple]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(sql, params)
            return await cur.fetchall()

    @staticmethod
    def _to_results(rows, mode: SearchMode) -> list[SearchResult]:
        return [
            SearchResult(
                chunk=Chunk(
                    document_id=document_id,
                    index=chunk_index,
                    content=text,
                    metadata=metadata or {},
                )
            ).with_score(mode.value, float(score))
            for document_id, chunk_index, text, metadata, score in rows
        ]

    def _vector_table_name(self) -> str:
        # Drop any "org/" prefix (e.g. "sentence-transformers/all-MiniLM-L6-v2").
        model = self.embedder.model.rsplit("/", 1)[-1]
        name = f"{model}_{self.embedder.dims}".lower()
        name = re.sub(r"[^a-z0-9_]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return name[:63]  # max table name limit
