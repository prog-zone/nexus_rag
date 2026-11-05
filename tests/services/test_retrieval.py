"""Retrieval-service behaviour: sub-query normalisation and context assembly."""

from types import SimpleNamespace

import pytest
from qdrant_client import models

from app.services.retrieval import RetrievalService, retrieval_service


class Point(SimpleNamespace):
    """Stand-in for a Qdrant ScoredPoint."""

    def __init__(self, text: str, score: float, doc_name: str = "acme.pdf", doc_id: str = "doc-1"):
        super().__init__(
            payload={"text": text, "doc_name": doc_name, "doc_id": doc_id},
            score=score,
        )


@pytest.fixture()
def service():
    return RetrievalService()


@pytest.fixture()
def quiet_embeddings(monkeypatch):
    async def fake_dense(text):
        return [1.0, 0.0]

    async def fake_sparse(text):
        return models.SparseVector(indices=[0], values=[1.0])

    monkeypatch.setattr("app.services.retrieval.embedding_service.generate_query_embedding", fake_dense)
    monkeypatch.setattr("app.services.retrieval.sparse_embedder.embed_query", fake_sparse)


# --------------------------------------------------------------------------- #
# sub_queries normalisation
# --------------------------------------------------------------------------- #

def test_sub_queries_from_a_string_is_not_split_into_characters(service):
    """A bare string used to be iterated character by character."""
    result = service._sub_queries_from({"sub_queries": "indemnity clause 9b"}, "original question")

    assert result == ["indemnity clause 9b"]


@pytest.mark.parametrize("raw", [[], None, "", ["   "], [None, 7]])
def test_sub_queries_fall_back_to_the_original_question(service, raw):
    """An empty or unusable list used to skip every search."""
    assert service._sub_queries_from({"sub_queries": raw}, "original question") == ["original question"]


def test_sub_queries_are_trimmed_deduped_and_capped(service):
    understood = {
        "sub_queries": ["  clause 9b ", "clause 9b", "indemnity", "", "termination", "payment", "extra"]
    }

    assert service._sub_queries_from(understood, "fallback") == [
        "clause 9b", "indemnity", "termination"
    ]


def test_sub_queries_tolerate_a_missing_key(service):
    assert service._sub_queries_from({}, "original question") == ["original question"]


# --------------------------------------------------------------------------- #
# context assembly
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_chat_history_is_ranked_across_sub_queries(service, monkeypatch, quiet_embeddings):
    """The weak hit of the first sub-query must not displace the strong one."""
    async def fake_understand(query, recent_messages):
        return {"sub_queries": ["alpha", "beta"], "query_type": "new_question", "doc_hint": None}

    async def fake_search_documents(*args, **kwargs):
        return []

    history = iter([[Point("weak exchange", 0.10)], [Point("strong exchange", 0.90)]])

    async def fake_search_chat_memory(dense, sparse, chat_id, user_id, source):
        if source == "chat_history":
            return next(history)
        return []

    async def fake_rerank(query, documents, top_k=3):
        return []

    monkeypatch.setattr(service, "_understand_query", fake_understand)
    monkeypatch.setattr("app.services.retrieval.qdrant_service.search_documents", fake_search_documents)
    monkeypatch.setattr("app.services.retrieval.qdrant_service.search_chat_memory", fake_search_chat_memory)
    monkeypatch.setattr("app.services.retrieval.reranker_service.rerank", fake_rerank)

    result = await service.retrieve(query="q", case_id="c", chat_id="ch", user_id="u")

    assert [chunk["text"] for chunk in result["chat_history_chunks"]] == [
        "strong exchange", "weak exchange"
    ]


@pytest.mark.asyncio
async def test_pastes_are_deduped_before_reranking(service, monkeypatch, quiet_embeddings):
    """Two sub-queries returning the same paste must not send it twice."""
    async def fake_understand(query, recent_messages):
        return {"sub_queries": ["alpha", "beta"], "query_type": "new_question", "doc_hint": None}

    async def fake_search_documents(*args, **kwargs):
        return []

    async def fake_search_chat_memory(dense, sparse, chat_id, user_id, source):
        if source == "inline_paste":
            return [Point("pasted clause", 0.8)]
        return []

    async def fake_rerank(query, documents, top_k=3):
        return [{"index": index, "text": text, "score": 0.9} for index, text in enumerate(documents)]

    monkeypatch.setattr(service, "_understand_query", fake_understand)
    monkeypatch.setattr("app.services.retrieval.qdrant_service.search_documents", fake_search_documents)
    monkeypatch.setattr("app.services.retrieval.qdrant_service.search_chat_memory", fake_search_chat_memory)
    monkeypatch.setattr("app.services.retrieval.reranker_service.rerank", fake_rerank)

    captured = {}
    real_rerank = fake_rerank

    async def spy_rerank(query, documents, top_k=3):
        captured["documents"] = list(documents)
        return await real_rerank(query, documents, top_k)

    monkeypatch.setattr("app.services.retrieval.reranker_service.rerank", spy_rerank)

    result = await service.retrieve(query="q", case_id="c", chat_id="ch", user_id="u")

    assert captured["documents"] == ["pasted clause"]
    assert [chunk["text"] for chunk in result["document_chunks"]] == ["pasted clause"]


@pytest.mark.asyncio
async def test_chunks_without_text_are_dropped(service):
    """An empty payload text carries no context and must not reach the prompt."""
    assert service._dedupe_by_text([
        {"text": "", "score": 0.9},
        {"text": "   ", "score": 0.8},
        {"text": "real chunk", "score": 0.7},
    ]) == [{"text": "real chunk", "score": 0.7}]


@pytest.mark.asyncio
async def test_empty_question_still_searches_with_the_original_query(service, monkeypatch, quiet_embeddings):
    """An empty sub-query list used to answer "not found" without searching."""
    async def fake_understand(query, recent_messages):
        return {"sub_queries": [], "query_type": "new_question", "doc_hint": None}

    searched = []

    async def fake_search_documents(dense, sparse, case_id, user_id, doc_hint=None, **kwargs):
        searched.append((case_id, doc_hint))
        return []

    async def fake_search_chat_memory(*args, **kwargs):
        return []

    monkeypatch.setattr(service, "_understand_query", fake_understand)
    monkeypatch.setattr("app.services.retrieval.qdrant_service.search_documents", fake_search_documents)
    monkeypatch.setattr("app.services.retrieval.qdrant_service.search_chat_memory", fake_search_chat_memory)

    result = await service.retrieve(query="what is clause 9b?", case_id="c", chat_id="ch", user_id="u")

    assert searched == [("c", None)]
    assert result["sub_queries"] == ["what is clause 9b?"]


@pytest.mark.asyncio
async def test_a_failing_search_degrades_to_no_context(service, monkeypatch, quiet_embeddings):
    async def fake_understand(query, recent_messages):
        return {"sub_queries": ["alpha"], "query_type": "new_question", "doc_hint": None}

    async def failing(*args, **kwargs):
        raise RuntimeError("qdrant unavailable")

    monkeypatch.setattr(service, "_understand_query", fake_understand)
    monkeypatch.setattr("app.services.retrieval.qdrant_service.search_documents", failing)
    monkeypatch.setattr("app.services.retrieval.qdrant_service.search_chat_memory", failing)

    result = await service.retrieve(query="q", case_id="c", chat_id="ch", user_id="u")

    assert result["has_relevant_context"] is False
    assert result["document_chunks"] == []


def test_module_singleton_is_exported():
    assert isinstance(retrieval_service, RetrievalService)
