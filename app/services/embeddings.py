import voyageai
from typing import cast
from app.core.config import settings
from app.core.logger import log


class EmbeddingService:
    def __init__(self):
        self.client = voyageai.AsyncClient(api_key=settings.VOYAGE_API_KEY)  # type: ignore
        self.model = settings.VOYAGE_EMBEDDING_MODEL

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embeddings via Voyage AI voyage-law-2."""
        response = await self.client.embed(
            texts=texts,
            model=self.model,
            input_type="document"
        )
        log.info("embeddings_generated", count=len(texts), model=self.model)
        return cast(list[list[float]], response.embeddings)

    async def generate_query_embedding(self, query: str) -> list[float]:
        response = await self.client.embed(
            texts=[query],
            model=self.model,
            input_type="query"
        )
        return cast(list[float], response.embeddings[0])


embedding_service = EmbeddingService()