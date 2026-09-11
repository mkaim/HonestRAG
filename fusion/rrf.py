import heapq

from models import ChunkID, RankFusion, SearchResult


class RRF(RankFusion):
    """Reciprocal Rank Fusion."""

    SCORE_FIELD = "rrf"

    def __init__(self, k: int = 60):
        self.k = k

    def merge(
        self, rankings: list[list[SearchResult]], limit: int
    ) -> list[SearchResult]:
        merged: dict[ChunkID, SearchResult] = {}
        rrf_scores: dict[ChunkID, float] = {}

        for ranking in rankings:
            for sr in ranking:
                out = merged.setdefault(sr.chunk.id, SearchResult(chunk=sr.chunk))
                for name, value in sr.scores.items():
                    out = out.with_score(name, value)
                merged[sr.chunk.id] = out
                rrf_scores.setdefault(sr.chunk.id, 0.0)

        for ranking in rankings:
            for rank, sr in enumerate(ranking, start=1):
                rrf_scores[sr.chunk.id] += 1.0 / (rank + self.k)

        merged = [
            sr.with_score(self.SCORE_FIELD, rrf_scores[chunk_id])
            for chunk_id, sr in merged.items()
        ]

        return heapq.nlargest(limit, merged, key=lambda sr: sr.score(self.SCORE_FIELD))
