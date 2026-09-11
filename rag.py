import asyncio

from models import CrossEncoder, RagBase, RagDb, RankFusion, SearchMode, SearchResult


class Rag(RagBase):
    MODES = (SearchMode.BM25, SearchMode.SEMANTIC)

    def __init__(
        self,
        db: RagDb,
        rank_fusion: RankFusion,
        cross_encoder: CrossEncoder | None = None,
        retrieval_limit: int = 100,
        rf_limit: int = 50,
        cross_encoder_limit: int = 10,
    ):
        self.db = db
        self.rank_fusion = rank_fusion
        self.cross_encoder = cross_encoder
        self.retrieval_limit = retrieval_limit
        self.rf_limit = rf_limit
        self.cross_encoder_limit = cross_encoder_limit

    async def search(self, queries: list[str]) -> list[list[SearchResult]]:
        # results_by_mode[i] is a list (per query) of ranked SearchResults.
        results_by_mode = await asyncio.gather(
            *(
                self.db.search(mode, queries, self.retrieval_limit)
                for mode in self.MODES
            )
        )

        merged_per_query = [
            self.rank_fusion.merge(list(rankings), self.rf_limit)
            for rankings in zip(*results_by_mode, strict=True)
        ]

        if self.cross_encoder is None:
            return merged_per_query

        return await asyncio.gather(
            *(
                self.cross_encoder.score(query, merged, self.cross_encoder_limit)
                for query, merged in zip(queries, merged_per_query, strict=True)
            )
        )
