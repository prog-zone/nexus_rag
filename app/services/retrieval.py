import json
import asyncio
from openai import AsyncOpenAI
from app.core.logger import log
from app.core.config import settings
from app.services.qdrant import qdrant_service
from app.services.embeddings import embedding_service
from app.services.reranker import reranker_service
from app.tasks.ingestion import _build_bm25_sparse_vector
from qdrant_client.models import SparseVector


RELEVANCE_THRESHOLD = 0.3

class RetrievalService:

    async def _understand_query(self, query: str, recent_messages: list) -> dict:
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        history_text = ""
        if recent_messages:
            history_text = "\n".join([
                f"{m['role'].upper()}: {m['content']}"
                for m in recent_messages[-5:]
            ])

        prompt = f"""You are a legal query analyzer. Extract structured search intent from the user's query.
            Recent conversation:
            {history_text}

            User query: {query}

            Respond ONLY with a JSON object, no markdown, no explanation:
            {{
                "sub_queries": ["precise search query 1", "precise search query 2"],
                "query_type": "simple | comparison | followup",
                "entities": ["clause name", "party name", "section number"]
            }}

            Rules:
            - For simple queries: one sub_query, same as the core legal entity
            - For comparisons across documents: one sub_query per document/topic
            - For follow-ups: include relevant context from conversation
            - Keep sub_queries short and precise, focused on legal entities
            - Maximum 3 sub_queries"""

        try:
            response = await client.chat.completions.create(
                model=settings.OPENAI_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200
            )
            text = response.choices[0].message.content or ""
            result = json.loads(text)
            log.info(
                "query_understood",
                query_type=result.get("query_type"),
                sub_queries=result.get("sub_queries"),
                entities=result.get("entities")
            )
            return result
        except Exception:
            log.warning("query_understanding_failed_falling_back", query=query)
            return {"sub_queries": [query], "query_type": "simple", "entities": []}

    async def retrieve(
        self,
        query: str,
        case_id: str,
        chat_id: str,
        user_id: str,
        recent_messages: list = [],
        top_k: int = 3
    ) -> dict:
        understood = await self._understand_query(query, recent_messages)
        sub_queries = understood.get("sub_queries", [query])

        all_doc_results = []
        all_paste_results = []
        all_history_results = []

        for sub_query in sub_queries:
            dense_vector = embedding_service.generate_query_embedding(sub_query)
            sparse_vector = SparseVector(**_build_bm25_sparse_vector(sub_query).__dict__)

            doc_results, paste_results, history_results = await asyncio.gather(
                self._search_documents(dense_vector, sparse_vector, case_id, user_id),
                self._search_chat_memory(dense_vector, sparse_vector, chat_id, "inline_paste"),
                self._search_chat_memory(dense_vector, sparse_vector, chat_id, "chat_history"),
            )

            all_doc_results.extend(doc_results)
            all_paste_results.extend(paste_results)
            all_history_results.extend(history_results)

        seen = set()
        unique_doc_results = []
        for r in all_doc_results:
            if r["text"] not in seen:
                seen.add(r["text"])
                unique_doc_results.append(r)

        log.info(
            "retrieval_raw_results",
            sub_queries=len(sub_queries),
            docs=len(unique_doc_results),
            pastes=len(all_paste_results),
            history=len(all_history_results)
        )

        chunks_to_rerank = unique_doc_results + all_paste_results
        reranked_chunks = []

        if chunks_to_rerank:
            texts = [c["text"] for c in chunks_to_rerank]
            reranked = reranker_service.rerank(query=query, documents=texts, top_k=top_k)

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

        return {
            "document_chunks": reranked_chunks,
            "chat_history_chunks": all_history_results[:3],
            "has_relevant_context": len(reranked_chunks) > 0,
            "query_type": understood.get("query_type", "simple"),
            "sub_queries": sub_queries
        }

    async def _search_documents(
        self,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        case_id: str,
        user_id: str
    ) -> list[dict]:
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
        if not retrieval_result["has_relevant_context"]:
            return ""

        sections = []

        if retrieval_result["document_chunks"]:
            sections.append("[Retrieved Context]")
            for chunk in retrieval_result["document_chunks"]:
                source_label = f"[Source: {chunk['doc_name']}]"
                sections.append(f"{source_label}\n{chunk['text']}")

        if retrieval_result["chat_history_chunks"]:
            sections.append("\n[Relevant Past Exchanges]")
            for chunk in retrieval_result["chat_history_chunks"]:
                sections.append(chunk["text"])

        return "\n\n".join(sections)


retrieval_service = RetrievalService()