import threading

from fastembed import SparseTextEmbedding
import asyncio
from qdrant_client.models import SparseVector
from app.core.logger import log

_model: SparseTextEmbedding | None = None
_model_lock = threading.Lock()


def _get_model() -> SparseTextEmbedding:
    """Load the BM25 model on first use.

    Constructing it at import time downloaded the model on every process start
    (API, worker, test collection), which made imports slow and network-bound.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _model


class SparseEmbedderService:

    async def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        """Batch sparse embeddings for ingestion."""
        model = _get_model()
        results = await asyncio.to_thread(lambda: list(model.embed(texts)))
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
        model = _get_model()
        result = await asyncio.to_thread(lambda: list(model.query_embed(query))[0])
        return SparseVector(
            indices=result.indices.tolist(),
            values=result.values.tolist()
        )


sparse_embedder = SparseEmbedderService()
