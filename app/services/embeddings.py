from fastembed import TextEmbedding, SparseTextEmbedding
from typing import List, Dict

class EmbeddingService:
    def __init__(self):
        # Dense model for semantic search
        self.dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        # Sparse model for keyword/exact matching
        self.sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

    def generate_hybrid_embeddings(self, texts: List[str]):
        dense_embeddings = list(self.dense_model.embed(texts))
        sparse_embeddings = list(self.sparse_model.embed(texts))
        
        return dense_embeddings, sparse_embeddings

embedding_service = EmbeddingService()