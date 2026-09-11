import httpx

from models import Embedder


class OpenAIEmbedder(Embedder):
    def __init__(self, base_url: str, api_key: str, model: str, dims: int):
        self.model = model
        self.dims = dims
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.post(
            "/embeddings",
            json={"model": self.model, "input": texts},
        )
        response.raise_for_status()
        return [row["embedding"] for row in response.json()["data"]]

    async def aclose(self) -> None:
        await self._client.aclose()
