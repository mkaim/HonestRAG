import heapq

from models import Chunk, ChunkID, RankFusion, SearchResult


class RRF(RankFusion):
    """Reciprocal Rank Fusion."""

    SCORE_FIELD = "rrf"

    def __init__(self, k: int = 60):
        self.k = k

    def merge(
        self, rankings: list[list[SearchResult]], limit: int
    ) -> list[SearchResult]:
        chunks: dict[ChunkID, Chunk] = {}
        scores_by_chunk: dict[ChunkID, dict[str, float]] = {}
        rrf_scores: dict[ChunkID, float] = {}

        for ranking in rankings:
            for rank, sr in enumerate(ranking, start=1):
                chunks.setdefault(sr.chunk.id, sr.chunk)
                # merge previous chunk's scores:
                scores_by_chunk.setdefault(sr.chunk.id, {}).update(sr.scores)
                # calculate this chunk's reciprocal-rank contribution for this ranking.
                rrf_scores[sr.chunk.id] = rrf_scores.get(sr.chunk.id, 0.0) + 1.0 / (
                    rank + self.k
                )

        results = [
            SearchResult(
                chunk=chunks[chunk_id],
                scores={
                    **scores_by_chunk[chunk_id],  # copy pervious scores of chunks
                    self.SCORE_FIELD: rrf_scores[chunk_id],
                },
            )
            for chunk_id in chunks
        ]

        return heapq.nlargest(limit, results, key=lambda sr: sr.score(self.SCORE_FIELD))
