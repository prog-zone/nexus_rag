import asyncio
from app.core.logger import log
from app.core.config import settings
from app.services.qdrant import qdrant_service
from app.services.embeddings import embedding_service
from app.services.reranker import reranker_service
from app.tasks.ingestion import _build_bm25_sparse_vector
from qdrant_client.models import SparseVector


# Minimum reranker score to include a chunk in context
# Below this → we don't guess, we tell user nothing found
RELEVANCE_THRESHOLD = 0.3


class RetrievalService:

    async def retrieve(
        self,
        query: str,
        case_id: str,
        chat_id: str,
        user_id: str,
        top_k: int = 3
    ) -> dict:
        """
        Full retrieval pipeline:
        1. Generate query embeddings (dense + sparse)
        2. Parallel search across all three sources
        3. Rerank combined results
        4. Return short, exact context
        """

        # ── Step 1: Query Embeddings ──────────────────────────────
        dense_vector = embedding_service.generate_query_embedding(query)
        sparse_vector = SparseVector(**_build_bm25_sparse_vector(query).__dict__)

        # ── Step 2: Parallel Retrieval ────────────────────────────
        doc_results, paste_results, history_results = await asyncio.gather(
            self._search_documents(dense_vector, sparse_vector, case_id, user_id),
            self._search_chat_memory(dense_vector, sparse_vector, chat_id, "inline_paste"),
            self._search_chat_memory(dense_vector, sparse_vector, chat_id, "chat_history"),
        )

        log.info(
            "retrieval_raw_results",
            docs=len(doc_results),
            pastes=len(paste_results),
            history=len(history_results)
        )

        # ── Step 3: Rerank Document + Paste Results ───────────────
        # Chat history exchange pairs are kept separate — not reranked
        # because they need to preserve conversational context
        chunks_to_rerank = doc_results + paste_results

        reranked_chunks = []
        if chunks_to_rerank:
            texts = [c["text"] for c in chunks_to_rerank]
            reranked = reranker_service.rerank(query=query, documents=texts, top_k=top_k)

            # Apply relevance threshold — fail loudly if nothing passes
            reranked_chunks = [
                {
                    "text": r["text"],
                    "score": r["score"],
                    "doc_name": chunks_to_rerank[r["index"]].get("doc_name", "unknown"),
                    "source": chunks_to_rerank[r["index"]].get("source", "document"),
                    "doc_id": chunks_to_rerank[r["index"]].get("doc_id", ""),
                }
                for r in reranked
                if r["score"] >= RELEVANCE_THRESHOLD
            ]

        # ── Step 4: Return structured context ─────────────────────
        return {
            "document_chunks": reranked_chunks,
            "chat_history_chunks": history_results[:3],
            "has_relevant_context": len(reranked_chunks) > 0
        }

    async def _search_documents(
        self,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        case_id: str,
        user_id: str
    ) -> list[dict]:
        """Search nexus_documents — runs in thread since qdrant client is sync."""
        try:
            results = await asyncio.to_thread(
                qdrant_service.search_documents,
                dense_vector, sparse_vector, case_id, user_id
            )
            return [
                {
                    "text": r.payload.get("text", ""),
                    "doc_name": r.payload.get("doc_name", "unknown"),
                    "doc_id": r.payload.get("doc_id", ""),
                    "source": "document",
                    "score": r.score
                }
                for r in results if r.payload
            ]
        except Exception:
            log.exception("document_search_failed")
            return []

    async def _search_chat_memory(
        self,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        chat_id: str,
        source: str
    ) -> list[dict]:
        """Search nexus_chat_memory for inline pastes or chat history."""
        try:
            results = await asyncio.to_thread(
                qdrant_service.search_chat_memory,
                dense_vector, sparse_vector, chat_id, source
            )
            return [
                {
                    "text": r.payload.get("text", ""),
                    "doc_name": r.payload.get("doc_name", source),
                    "doc_id": r.payload.get("doc_id", ""),
                    "source": source,
                    "score": r.score
                }
                for r in results if r.payload
            ]
        except Exception:
            log.exception("chat_memory_search_failed", source=source)
            return []

    def build_context_prompt(self, retrieval_result: dict) -> str:
        """
        Assembles the context string to inject into the LLM prompt.
        Short, exact, cited.
        """
        if not retrieval_result["has_relevant_context"]:
            return ""

        sections = []

        # Document + paste chunks (reranked, cited)
        if retrieval_result["document_chunks"]:
            sections.append("[Retrieved Context]")
            for chunk in retrieval_result["document_chunks"]:
                source_label = f"[Source: {chunk['doc_name']}]"
                sections.append(f"{source_label}\n{chunk['text']}")

        # Chat history exchanges (not reranked, kept as-is)
        if retrieval_result["chat_history_chunks"]:
            sections.append("\n[Relevant Past Exchanges]")
            for chunk in retrieval_result["chat_history_chunks"]:
                sections.append(chunk["text"])

        return "\n\n".join(sections)


retrieval_service = RetrievalService()