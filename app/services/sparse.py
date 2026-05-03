import asyncio
from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector
from app.core.logger import log


class SparseEmbedderService:

    def __init__(self):
        self.model = SparseTextEmbedding(
            model_name="Qdrant/bm25"
        )

    async def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        """Batch sparse embeddings for ingestion."""
        results = await asyncio.to_thread(lambda: list(self.model.embed(texts)))
        log.info("sparse_embeddings_generated", count=len(texts))
        return [
            SparseVector(
                indices=r.indices.tolist(),
                values=r.values.tolist()
            )
            for r in results
        ]

    async def embed_query(self, query: str) -> SparseVector:
        """Single sparse embedding for query time."""
        result = await asyncio.to_thread(lambda: list(self.model.query_embed(query))[0])
        return SparseVector(
            indices=result.indices.tolist(),
            values=result.values.tolist()
        )


sparse_embedder = SparseEmbedderService()