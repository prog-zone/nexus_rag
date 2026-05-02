from qdrant_client import QdrantClient, models
from app.core.config import settings
from app.core.logger import log
from typing import List


class QdrantService:
    def __init__(self):
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.documents_collection = "nexus_documents"
        self.chat_memory_collection = "nexus_chat_memory"

        # voyage-law-2 embedding dimension
        self.dense_dim = 1024

    def _vector_config(self):
        return {
            "dense": models.VectorParams(
                size=self.dense_dim,
                distance=models.Distance.COSINE
            )
        }

    def _sparse_config(self):
        # Qdrant native BM25 — no token limit, pure statistical term matching
        return {
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF  # enables BM25 scoring
            )
        }

    def ensure_collections(self):
        if not self.client.collection_exists(self.documents_collection):
            self.client.create_collection(
                collection_name=self.documents_collection,
                vectors_config=self._vector_config(),
                sparse_vectors_config=self._sparse_config()
            )
            log.info("qdrant_collection_created", name=self.documents_collection)

        if not self.client.collection_exists(self.chat_memory_collection):
            self.client.create_collection(
                collection_name=self.chat_memory_collection,
                vectors_config=self._vector_config(),
                sparse_vectors_config=self._sparse_config()
            )
            log.info("qdrant_collection_created", name=self.chat_memory_collection)

    def upsert_documents(self, points: List[models.PointStruct]):
        return self.client.upsert(
            collection_name=self.documents_collection,
            points=points
        )

    def upsert_chat_memory(self, points: List[models.PointStruct]):
        return self.client.upsert(
            collection_name=self.chat_memory_collection,
            points=points
        )

    def search_documents(
        self,
        dense_vector: list[float],
        sparse_vector: models.SparseVector,
        case_id: str,
        user_id: str,
        top_k: int = 5
    ) -> list[models.ScoredPoint]:
        """Hybrid search on nexus_documents filtered by case."""
        return self.client.query_points(
            collection_name=self.documents_collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=top_k * 2
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=top_k * 2
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="case_id",
                        match=models.MatchValue(value=case_id)
                    ),
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            ),
            limit=top_k
        ).points

    def search_chat_memory(
        self,
        dense_vector: list[float],
        sparse_vector: models.SparseVector,
        chat_id: str,
        source: str,
        top_k: int = 3
    ) -> list[models.ScoredPoint]:
        """Search nexus_chat_memory filtered by chat and source type."""
        return self.client.query_points(
            collection_name=self.chat_memory_collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=top_k * 2
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=top_k * 2
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="chat_id",
                        match=models.MatchValue(value=chat_id)
                    ),
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source)
                    )
                ]
            ),
            limit=top_k
        ).points


qdrant_service = QdrantService()