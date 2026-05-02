from fastembed import TextEmbedding, SparseTextEmbedding
from typing import List

class EmbeddingService:
    def __init__(self):
        self._dense_model: TextEmbedding | None = None
        self._sparse_model: SparseTextEmbedding | None = None

    @property
    def dense_model(self) -> TextEmbedding:
        if self._dense_model is None:
            self._dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return self._dense_model

    @property
    def sparse_model(self) -> SparseTextEmbedding:
        if self._sparse_model is None:
            self._sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
        return self._sparse_model

    def generate_hybrid_embeddings(self, texts: List[str]):
        dense_embeddings = list(self.dense_model.embed(texts))
        sparse_embeddings = list(self.sparse_model.embed(texts))
        return dense_embeddings, sparse_embeddings

embedding_service = EmbeddingService()