"""Vector-store behaviour: collection bootstrap, filtered search and deletion."""

import uuid

import pytest
from qdrant_client import AsyncQdrantClient, models

from app.services.qdrant import QdrantService


def _point(case_id: str, text: str, doc_name: str, doc_id: str, user_id: str = "user-1"):
    return models.PointStruct(
        id=str(uuid.uuid4()),
        vector={"dense": [1.0, 0.0, 0.0, 0.0], "sparse": models.SparseVector(indices=[0], values=[1.0])},
        payload={
            "text": text,
            "doc_name": doc_name,
            "doc_id": doc_id,
            "case_id": case_id,
            "user_id": user_id,
            "source": "document",
        },
    )


@pytest.fixture()
def service(in_memory_qdrant):
    svc = QdrantService()
    svc.dense_dim = 4
    svc.__dict__["client"] = in_memory_qdrant
    return svc


@pytest.mark.asyncio
async def test_ensure_collections_creates_both_collections(service):
    """A fresh Qdrant must end up with both collections.

    ``collection_exists`` is a coroutine on the async client: without the await
    the branch was never taken and the collections were never created, so the
    first ingestion died with "Collection nexus_documents not found".
    """
    await service.ensure_collections()

    assert await service.client.collection_exists(service.documents_collection)
    assert await service.client.collection_exists(service.chat_memory_collection)


@pytest.mark.asyncio
async def test_ensure_collections_accepts_a_repeated_call(service):
    await service.ensure_collections()
    await service.ensure_collections()  # must not raise "already exists"

    assert await service.client.collection_exists(service.documents_collection)


@pytest.mark.asyncio
async def test_upsert_succeeds_after_bootstrap(service):
    """The bootstrap must leave the collection usable, not merely present."""
    await service.ensure_collections()

    await service.upsert_documents([_point("case-1", "indemnity clause", "acme.pdf", "doc-1")])

    assert await service.count_points(service.documents_collection, "case-1") == 1


@pytest.mark.asyncio
async def test_search_documents_returns_the_case_chunks(service):
    await service.ensure_collections()
    await service.upsert_documents([
        _point("case-1", "indemnity clause text", "acme.pdf", "doc-1"),
        _point("case-2", "other case text", "other.pdf", "doc-2"),
    ])

    points = await service.search_documents(
        [1.0, 0.0, 0.0, 0.0],
        models.SparseVector(indices=[0], values=[1.0]),
        case_id="case-1",
        user_id="user-1",
    )

    assert [p.payload["doc_name"] for p in points] == ["acme.pdf"]


@pytest.mark.asyncio
async def test_search_documents_retries_without_a_failing_doc_hint(service, monkeypatch):
    """A doc_hint the server cannot evaluate must not turn into "no results"."""
    await service.ensure_collections()
    await service.upsert_documents([_point("case-1", "indemnity clause text", "acme.pdf", "doc-1")])

    calls = []
    real_query_points = service.client.query_points

    async def flaky_query_points(**kwargs):
        calls.append(kwargs)
        filter_has_hint = any(
            getattr(condition, "key", None) == "doc_name"
            for condition in kwargs["query_filter"].must
        )
        if filter_has_hint:
            raise ValueError("Bad request: field doc_name has no full-text index")
        return await real_query_points(**kwargs)

    monkeypatch.setattr(service.client, "query_points", flaky_query_points)

    points = await service.search_documents(
        [1.0, 0.0, 0.0, 0.0],
        models.SparseVector(indices=[0], values=[1.0]),
        case_id="case-1",
        user_id="user-1",
        doc_hint="acme.pdf",
    )

    assert len(calls) == 2
    assert points and points[0].payload["doc_name"] == "acme.pdf"


@pytest.mark.asyncio
async def test_search_documents_without_a_hint_does_not_retry(service, monkeypatch):
    """Only the hint path is allowed a retry; a real failure must surface."""
    await service.ensure_collections()

    calls = []

    async def failing_query_points(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("qdrant is down")

    monkeypatch.setattr(service.client, "query_points", failing_query_points)

    with pytest.raises(RuntimeError):
        await service.search_documents(
            [1.0, 0.0, 0.0, 0.0],
            models.SparseVector(indices=[0], values=[1.0]),
            case_id="case-1",
            user_id="user-1",
        )

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_delete_by_doc_id_also_clears_inline_pastes(service):
    """Inline pastes live in the chat-memory collection under the same doc_id."""
    await service.ensure_collections()
    await service.upsert_documents([_point("case-1", "uploaded text", "acme.pdf", "doc-1")])
    await service.upsert_chat_memory([
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector={"dense": [1.0, 0.0, 0.0, 0.0], "sparse": models.SparseVector(indices=[0], values=[1.0])},
            payload={
                "text": "pasted text",
                "doc_id": "doc-2",
                "case_id": "case-1",
                "chat_id": "chat-1",
                "user_id": "user-1",
                "source": "inline_paste",
            },
        )
    ])

    await service.delete_by_doc_id("doc-2")

    assert await service.count_points(service.chat_memory_collection, "case-1") == 0
    assert await service.count_points(service.documents_collection, "case-1") == 1


@pytest.mark.asyncio
async def test_delete_by_case_id_clears_both_collections(service):
    await service.ensure_collections()
    await service.upsert_documents([_point("case-1", "uploaded text", "acme.pdf", "doc-1")])
    await service.upsert_chat_memory([
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector={"dense": [1.0, 0.0, 0.0, 0.0], "sparse": models.SparseVector(indices=[0], values=[1.0])},
            payload={"text": "history", "case_id": "case-1", "chat_id": "chat-1", "user_id": "user-1",
                     "source": "chat_history"},
        )
    ])

    await service.delete_by_case_id("case-1")

    assert await service.count_points(service.documents_collection, "case-1") == 0
    assert await service.count_points(service.chat_memory_collection, "case-1") == 0


@pytest.mark.asyncio
async def test_client_is_created_lazily(monkeypatch):
    """Importing the module must not open a connection to Qdrant."""
    service = QdrantService()
    assert "client" not in service.__dict__

    monkeypatch.setattr("app.services.qdrant.settings.QDRANT_HOST", "qdrant.internal")
    assert isinstance(service.client, AsyncQdrantClient)
    assert "client" in service.__dict__
