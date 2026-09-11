import asyncio

from sentence_transformers import SentenceTransformer

from models import Embedder


class SenTranEmbedder(Embedder):
    def __init__(self, model: str, batch_size: int = 64):
        self.model = model
        self.batch_size = batch_size
        self.transformer = SentenceTransformer(model)
        # get_sentence_embedding_dimension was renamed to get_embedding_dimension;
        # fall back to the old name for sentence-transformers versions that
        # predate the rename.
        get_dim = getattr(
            self.transformer,
            "get_embedding_dimension",
            self.transformer.get_sentence_embedding_dimension,
        )
        self.dims = get_dim()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = await asyncio.to_thread(
            self.transformer.encode,
            texts,
            normalize_embeddings=True,
            batch_size=self.batch_size,
        )
        return embeddings.tolist()
