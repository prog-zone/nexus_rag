import json
import asyncio
from openai import AsyncOpenAI
from app.core.logger import log
from app.core.config import settings
from app.services.qdrant import qdrant_service
from app.services.embeddings import embedding_service
from app.services.sparse import sparse_embedder
from app.services.reranker import reranker_service


RELEVANCE_THRESHOLD = 0.3
QDRANT_FETCH_K = 15
PRE_FILTER_K = 8
RERANK_TOP_K = 3


class RetrievalService:

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def _understand_query(self, query: str, recent_messages: list) -> dict:
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
                        "exact_entity": "the specific clause, section, or legal term being asked about (e.g. 'clause 9b-a', 'indemnity section', 'termination clause') or null if none",
                        "doc_hint": "the specific document name mentioned by the user (e.g. 'acme_contract_2024.pdf', 'xyz_agreement') or null if no specific document mentioned",
                        "query_type": "new_question | follow_up | comparison",
                        "sub_queries": ["precise search query 1", "precise search query 2"]
                    }}
                    Rules:
                    - exact_entity: extract the most specific legal term or clause reference from the query
                    - doc_hint: only set if the user explicitly names or refers to a specific document, otherwise null
                    - query_type:
                        * new_question — standalone question not referencing prior conversation
                        * follow_up — references something discussed earlier ("what about...", "and the...", "also check...")
                        * comparison — asks to compare across multiple documents or clauses
                    - sub_queries:
                        * new_question: one sub_query focused on the exact_entity
                        * follow_up: include context from recent conversation in the sub_query
                        * comparison: one sub_query per document or clause being compared
                        * Maximum 3 sub_queries, keep them short and precise"""

        try:
            response = await self.client.chat.completions.create(
                model=settings.OPENAI_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],  # type: ignore
                temperature=0,
                max_tokens=300
            )
            text = response.choices[0].message.content or ""
            result = json.loads(text)
            log.info(
                "query_understood",
                query_type=result.get("query_type"),
                exact_entity=result.get("exact_entity"),
                doc_hint=result.get("doc_hint"),
                sub_queries=result.get("sub_queries"),
            )
            return result
        except Exception:
            log.warning("query_understanding_failed_falling_back", query=query)
            return {
                "exact_entity": None,
                "doc_hint": None,
                "query_type": "new_question",
                "sub_queries": [query]
            }

    async def retrieve(
        self,
        query: str,
        case_id: str,
        chat_id: str,
        user_id: str,
        recent_messages: list | None = None,
    ) -> dict:
        if recent_messages is None:
            recent_messages = []

        understood = await self._understand_query(query, recent_messages)
        sub_queries = understood.get("sub_queries", [query])
        doc_hint = understood.get("doc_hint")

        all_doc_results = []
        all_paste_results = []
        all_history_results = []

        for sub_query in sub_queries:
            dense_vector = embedding_service.generate_query_embedding(sub_query)
            sparse_vector = sparse_embedder.embed_query(sub_query)

            doc_results, paste_results, history_results = await asyncio.gather(
                self._search_documents(dense_vector, sparse_vector, case_id, user_id, doc_hint),
                self._search_chat_memory(dense_vector, sparse_vector, chat_id, user_id, "inline_paste"),
                self._search_chat_memory(dense_vector, sparse_vector, chat_id, user_id, "chat_history"),
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
            doc_hint=doc_hint,
            docs=len(unique_doc_results),
            pastes=len(all_paste_results),
            history=len(all_history_results)
        )

        chunks_to_rerank = unique_doc_results + all_paste_results
        reranked_chunks = []

        if chunks_to_rerank:
            chunks_to_rerank.sort(key=lambda c: c["score"], reverse=True)
            chunks_to_rerank = chunks_to_rerank[:PRE_FILTER_K]

            log.info("pre_filter_applied", before=len(unique_doc_results + all_paste_results), after=len(chunks_to_rerank))

            texts = [c["text"] for c in chunks_to_rerank]
            reranked = reranker_service.rerank(query=query, documents=texts, top_k=RERANK_TOP_K)

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

            log.info("reranking_complete", final_chunks=len(reranked_chunks))

        return {
            "document_chunks": reranked_chunks,
            "chat_history_chunks": all_history_results[:3],
            "has_relevant_context": len(reranked_chunks) > 0,
            "query_type": understood.get("query_type", "new_question"),
            "exact_entity": understood.get("exact_entity"),
            "doc_hint": doc_hint,
            "sub_queries": sub_queries
        }

    async def _search_documents(
        self,
        dense_vector: list[float],
        sparse_vector,
        case_id: str,
        user_id: str,
        doc_hint: str | None = None
    ) -> list[dict]:
        try:
            results = await asyncio.to_thread(
                qdrant_service.search_documents,
                dense_vector, sparse_vector, case_id, user_id,
                top_k=QDRANT_FETCH_K, doc_hint=doc_hint
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
        sparse_vector,
        chat_id: str,
        user_id: str,
        source: str
    ) -> list[dict]:
        try:
            results = await asyncio.to_thread(
                qdrant_service.search_chat_memory,
                dense_vector, sparse_vector, chat_id, user_id, source
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