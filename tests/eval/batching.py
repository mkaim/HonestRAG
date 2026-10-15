"""Caps how many queries run concurrently against Postgres.

Eval fires up to RETRIEVAL_LIMIT-many questions at once via
asyncio.gather (25-480 depending on the corpus/test). Confirmed by trial
(not a database-side resource limit - ruled out pool size, event loop
scope, parallel workers, and open transactions one at a time before
finding this): with enough concurrent tasks scheduled on one event loop,
psycopg_pool's own background connection-management coroutines (also
tasks on that same loop) stop getting scheduled often enough to keep the
pool healthy under load, and the pool silently stops growing well short
of its max_size - requests queue up and eventually hit the client-side
PoolTimeout, which looks like a database problem but is actually event
loop contention. Batching into groups of ~10 and awaiting each batch
fully before starting the next reproducibly fixes it; the live app never
hits this since it only ever searches a handful of sub-questions at once.

BatchedRagDb wraps a RagDb and splits `queries` into batches of at most
`batch_size` before delegating, each batch awaited in full before the
next starts.
"""

from models import Chunk, Document, RagDb, SearchMode, SearchResult


class BatchedRagDb(RagDb):
    def __init__(self, db: RagDb, batch_size: int = 8):
        self.db = db
        self.batch_size = batch_size

    @property
    def embedder(self):
        return self.db.embedder

    async def search(
        self, mode: SearchMode, queries: list[str], limit: int
    ) -> list[list[SearchResult]]:
        results: list[list[SearchResult]] = []
        for i in range(0, len(queries), self.batch_size):
            batch = queries[i : i + self.batch_size]
            results.extend(await self.db.search(mode, batch, limit))
        return results

    async def add(self, document: Document, chunks: list[Chunk]) -> None:
        await self.db.add(document, chunks)
