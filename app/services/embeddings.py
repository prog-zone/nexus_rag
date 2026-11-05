import voyageai
from functools import cached_property
from typing import cast
from app.core.config import settings
from app.core.logger import log


class EmbeddingService:
    def __init__(self):
        self.model = settings.VOYAGE_EMBEDDING_MODEL

    @cached_property
    def client(self) -> voyageai.AsyncClient:  # type: ignore
        """Create the Voyage client on first use.

        Building it at import time made every module that touches embeddings
        require a live ``VOYAGE_API_KEY`` just to be imported.
        """
        if not settings.VOYAGE_API_KEY:
            raise RuntimeError("VOYAGE_API_KEY is not configured")
        return voyageai.AsyncClient(api_key=settings.VOYAGE_API_KEY)  # type: ignore

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embeddings via Voyage AI voyage-law-2."""
        response = await self.client.embed(
            texts=texts,
            model=self.model,
            input_type="document"
        )
        log.info("embeddings_generated", count=len(texts), model=self.model)
        embeddings = cast(list[list[float]], response.embeddings)
        if len(embeddings) != len(texts):
            raise ValueError(
                f"embedding provider returned {len(embeddings)} vectors for {len(texts)} texts"
            )
        return embeddings

    async def generate_query_embedding(self, query: str) -> list[float]:
        response = await self.client.embed(
            texts=[query],
            model=self.model,
            input_type="query"
        )
        return cast(list[float], response.embeddings[0])


embedding_service = EmbeddingService()
