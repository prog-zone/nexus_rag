from qdrant_client import QdrantClient, models
from app.core.config import settings
from app.core.logger import log
from typing import List

class QdrantService:
    def __init__(self):
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.collection_name = "nexus_documents"

    def ensure_collection(self):
        """Initializes the collection with both Dense and Sparse configurations."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=384, # BGE-Small dimensions
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=True))
                }
            )
            log.info("qdrant_collection_initialized", name=self.collection_name)

    def upsert_points(self, points: List[models.PointStruct]):
        return self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

qdrant_service = QdrantService()