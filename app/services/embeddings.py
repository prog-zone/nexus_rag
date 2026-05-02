import voyageai
from app.core.config import settings
from app.core.logger import log


class EmbeddingService:
    def __init__(self):
        self.client = voyageai.Client(api_key=settings.VOYAGE_API_KEY)
        self.model = settings.VOYAGE_EMBEDDING_MODEL

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embeddings via Voyage AI voyage-law-2."""
        response = self.client.embed(
            texts=texts,
            model=self.model,
            input_type="document"
        )
        log.info("embeddings_generated", count=len(texts), model=self.model)
        return response.embeddings

    def generate_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for a query (different input_type for better retrieval)."""
        response = self.client.embed(
            texts=[query],
            model=self.model,
            input_type="query"
        )
        return response.embeddings[0]


embedding_service = EmbeddingService()