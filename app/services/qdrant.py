from functools import cached_property
from typing import List

from qdrant_client import AsyncQdrantClient, models

from app.core.config import settings
from app.core.logger import log


class QdrantService:
    def __init__(self):
        self.documents_collection = "nexus_documents"
        self.chat_memory_collection = "nexus_chat_memory"
        self.dense_dim = 1024

    @cached_property
    def client(self) -> AsyncQdrantClient:
        """Create the client on first use so imports never open a socket."""
        return AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

    def _vector_config(self):
        return {
            "dense": models.VectorParams(
                size=self.dense_dim,
                distance=models.Distance.COSINE
            )
        }

    def _sparse_config(self):
        return {
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        }

    async def _ensure_doc_name_text_index(self, collection_name: str) -> None:
        """``MatchText`` filters need a full-text index on ``doc_name``."""
        await self.client.create_payload_index(
            collection_name=collection_name,
            field_name="doc_name",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase=True,
            ),
            wait=True,
        )

    async def ensure_collections(self):
        for name in (self.documents_collection, self.chat_memory_collection):
            # ``collection_exists`` is a coroutine on the async client: awaiting it
            # is what makes the bootstrap actually run. Without the await the
            # condition was never truthy and no collection was ever created, so the
            # first ingestion failed with "Collection nexus_documents not found".
            if await self.client.collection_exists(name):
                continue
            await self.client.create_collection(
                collection_name=name,
                vectors_config=self._vector_config(),
                sparse_vectors_config=self._sparse_config()
            )
            await self._ensure_doc_name_text_index(name)
            log.info("qdrant_collection_created", name=name)

    async def upsert_documents(self, points: List[models.PointStruct]):
        return await self.client.upsert(
            collection_name=self.documents_collection,
            points=points
        )

    async def upsert_chat_memory(self, points: List[models.PointStruct]):
        return await self.client.upsert(
            collection_name=self.chat_memory_collection,
            points=points
        )

    async def search_documents(
        self,
        dense_vector: list[float],
        sparse_vector: models.SparseVector,
        case_id: str,
        user_id: str,
        top_k: int = 15,
        doc_hint: str | None = None
    ) -> list[models.ScoredPoint]:

        must_conditions = [
            models.FieldCondition(
                key="case_id",
                match=models.MatchValue(value=case_id)
            ),
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id)
            ),
        ]

        if doc_hint:
            must_conditions.append(
                models.FieldCondition(
                    key="doc_name",
                    match=models.MatchText(text=doc_hint)
                )
            )
            log.info("doc_hint_filter_applied", doc_hint=doc_hint)

        try:
            response = await self.client.query_points(
                collection_name=self.documents_collection,
                prefetch=[
                    models.Prefetch(query=dense_vector, using="dense", limit=top_k * 2),
                    models.Prefetch(query=sparse_vector, using="sparse", limit=top_k * 2)
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=models.Filter(must=list(must_conditions)),
                limit=top_k
            )
            return response.points
        except Exception:
            if not doc_hint:
                raise
            # A document hint the server cannot evaluate must not take the whole
            # search down: retrieval callers treat an exception as "no documents",
            # which silently answered "not found in the case documents" even though
            # the chunks were there.
            log.exception("doc_hint_filter_failed_retrying_without_hint", doc_hint=doc_hint)
            return await self.search_documents(
                dense_vector, sparse_vector, case_id, user_id,
                top_k=top_k, doc_hint=None
            )

    async def search_chat_memory(
        self,
        dense_vector: list[float],
        sparse_vector: models.SparseVector,
        chat_id: str,
        user_id: str,
        source: str,
        top_k: int = 3
    ) -> list[models.ScoredPoint]:
        response = await self.client.query_points(
            collection_name=self.chat_memory_collection,
            prefetch=[
                models.Prefetch(query=dense_vector, using="dense", limit=top_k * 2),
                models.Prefetch(query=sparse_vector, using="sparse", limit=top_k * 2)
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="chat_id", match=models.MatchValue(value=chat_id)),
                    models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                    models.FieldCondition(key="source", match=models.MatchValue(value=source))
                ]
            ),
            limit=top_k
        )
        return response.points

    async def delete_by_doc_id(self, doc_id: str):
        await self.client.delete(
            collection_name=self.documents_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id)
                        )
                    ]
                )
            )
        )
        # Inline pastes live in the chat-memory collection and carry the same doc_id.
        await self.client.delete(
            collection_name=self.chat_memory_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id)
                        )
                    ]
                )
            )
        )
        log.info("qdrant_doc_vectors_deleted", doc_id=doc_id)

    async def delete_by_chat_id(self, chat_id: str):
        await self.client.delete(
            collection_name=self.chat_memory_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="chat_id",
                            match=models.MatchValue(value=chat_id)
                        )
                    ]
                )
            )
        )
        log.info("qdrant_chat_vectors_deleted", chat_id=chat_id)

    async def delete_by_case_id(self, case_id: str):
        """Remove every vector that belongs to a case, from both collections."""
        for collection_name in (self.documents_collection, self.chat_memory_collection):
            await self.client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="case_id",
                                match=models.MatchValue(value=case_id)
                            )
                        ]
                    )
                )
            )
        log.info("qdrant_case_vectors_deleted", case_id=case_id)

    async def count_points(self, collection_name: str, case_id: str) -> int:
        """Number of vectors stored for a case (used by tests and diagnostics)."""
        result = await self.client.count(
            collection_name=collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="case_id",
                        match=models.MatchValue(value=case_id)
                    )
                ]
            ),
            exact=True,
        )
        return result.count


qdrant_service = QdrantService()
