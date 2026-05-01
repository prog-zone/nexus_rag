from typing import List
from qdrant_client import models
from app.services.qdrant import qdrant_service
from app.services.embeddings import embedding_service
from app.core.logger import log

class RetrievalService:
    def __init__(self):
        self.collection_name = "nexus_documents"

    async def search(self, query_text: str, limit: int = 5):
        """
        Performs a Stage 1 Hybrid Search using Reciprocal Rank Fusion (RRF).
        This combines semantic (dense) and keyword (sparse) results.
        """
        dense_vecs, sparse_vecs = embedding_service.generate_hybrid_embeddings([query_text])
        query_dense = dense_vecs[0].tolist()
        query_sparse = sparse_vecs[0]

        result = qdrant_service.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_dense,
                    using="dense",
                    limit=20,
                ),
                models.Prefetch(
                    query=query_sparse.as_object(),
                    using="sparse",
                    limit=20,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True
        )

        log.info("hybrid_search_complete", query=query_text, results_count=len(result.points))
        
        return result.points

retrieval_service = RetrievalService()