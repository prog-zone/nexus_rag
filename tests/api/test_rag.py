import io
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_create_case_success(logged_in_client):
    """Test that an authenticated user can create a case."""
    response = await logged_in_client.post("/api/v1/rag/cases", json={
        "title": "Smith vs Jones",
        "description": "Contract dispute case"
    })
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "Smith vs Jones"
    assert data["description"] == "Contract dispute case"
    assert data["status"] == "active"
    assert "id" in data


@pytest.mark.asyncio
async def test_create_case_unauthorized():
    """Test that an unauthenticated request cannot create a case."""
    from httpx import AsyncClient, ASGITransport
    from app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="https://testserver") as client:
        response = await client.post("/api/v1/rag/cases", json={"title": "Should Fail"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_create_case_title_too_short(logged_in_client):
    """Test that a case title below min_length is rejected."""
    response = await logged_in_client.post("/api/v1/rag/cases", json={
        "title": "X"
    })
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_cases(logged_in_client):
    """Test that listing cases returns the user's cases."""
    await logged_in_client.post("/api/v1/rag/cases", json={"title": "List Test Case"})
    response = await logged_in_client.get("/api/v1/rag/cases")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) >= 1


@pytest.mark.asyncio
async def test_get_case_success(logged_in_client):
    """Test retrieving a specific case by ID."""
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "Get Case Test"})
    case_id = create.json()["id"]

    response = await logged_in_client.get(f"/api/v1/rag/cases/{case_id}")
    assert response.status_code == 200
    assert response.json()["id"] == case_id


@pytest.mark.asyncio
async def test_get_case_not_found(logged_in_client):
    """Test that fetching a non-existent case returns 404."""
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = await logged_in_client.get(f"/api/v1/rag/cases/{fake_id}")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_update_case_success(logged_in_client):
    """Test updating a case title and status."""
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "Old Title"})
    case_id = create.json()["id"]

    response = await logged_in_client.patch(f"/api/v1/rag/cases/{case_id}", json={
        "title": "New Title",
        "status": "archived"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "New Title"
    assert data["status"] == "archived"


@pytest.mark.asyncio
async def test_delete_case_success(logged_in_client):
    """Test that a case can be deleted and is gone afterwards."""
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "Delete Me"})
    case_id = create.json()["id"]

    delete_response = await logged_in_client.delete(f"/api/v1/rag/cases/{case_id}")
    assert delete_response.status_code == 204

    get_response = await logged_in_client.get(f"/api/v1/rag/cases/{case_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_delete_case_not_found(logged_in_client):
    """Test that deleting a non-existent case returns 404."""
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = await logged_in_client.delete(f"/api/v1/rag/cases/{fake_id}")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_upload_document_success(logged_in_client):
    """Test that a document can be uploaded successfully."""
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "Doc Upload Case"})
    case_id = create.json()["id"]

    with patch("app.api.rag.s3_service.upload_file", new_callable=AsyncMock), \
         patch("app.api.rag.process_document_pipeline.kiq", new_callable=AsyncMock):
        response = await logged_in_client.post(
            f"/api/v1/rag/cases/{case_id}/documents",
            files={"file": ("test.pdf", io.BytesIO(b"fake pdf content"), "application/pdf")}
        )

    assert response.status_code == 201
    data = response.json()
    assert "document_id" in data
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_upload_document_case_not_found(logged_in_client):
    """Test uploading to a non-existent case returns 404."""
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = await logged_in_client.post(
        f"/api/v1/rag/cases/{fake_id}/documents",
        files={"file": ("test.pdf", io.BytesIO(b"content"), "application/pdf")}
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_upload_document_file_too_large(logged_in_client):
    """Test that uploading a file exceeding size limit returns 400."""
    from app.core.config import settings

    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "Size Limit Case"})
    case_id = create.json()["id"]

    oversized_content = b"x" * (settings.MAX_FILE_SIZE_MB * 1024 * 1024 + 1)

    response = await logged_in_client.post(
        f"/api/v1/rag/cases/{case_id}/documents",
        files={"file": ("big.pdf", io.BytesIO(oversized_content), "application/pdf")}
    )
    assert response.status_code == 400
    assert "maximum allowed size" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_document_count_limit(logged_in_client):
    """Test that uploading beyond MAX_DOCS_PER_CASE returns 400."""
    from app.core.config import settings

    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "Count Limit Case"})
    case_id = create.json()["id"]

    with patch("app.api.rag.s3_service.upload_file", new_callable=AsyncMock), \
         patch("app.api.rag.process_document_pipeline.kiq", new_callable=AsyncMock):
        for _ in range(settings.MAX_DOCS_PER_CASE):
            await logged_in_client.post(
                f"/api/v1/rag/cases/{case_id}/documents",
                files={"file": ("test.pdf", io.BytesIO(b"content"), "application/pdf")}
            )

        response = await logged_in_client.post(
            f"/api/v1/rag/cases/{case_id}/documents",
            files={"file": ("one_too_many.pdf", io.BytesIO(b"content"), "application/pdf")}
        )

    assert response.status_code == 400
    assert "maximum limit" in response.json()["detail"]


@pytest.mark.asyncio
async def test_list_documents(logged_in_client):
    """Test listing documents for a case."""
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "List Docs Case"})
    case_id = create.json()["id"]

    with patch("app.api.rag.s3_service.upload_file", new_callable=AsyncMock), \
         patch("app.api.rag.process_document_pipeline.kiq", new_callable=AsyncMock):
        await logged_in_client.post(
            f"/api/v1/rag/cases/{case_id}/documents",
            files={"file": ("test.pdf", io.BytesIO(b"content"), "application/pdf")}
        )

    response = await logged_in_client.get(f"/api/v1/rag/cases/{case_id}/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 1


@pytest.mark.asyncio
async def test_delete_document_success(logged_in_client):
    """Test that a document can be deleted and vectors are cleaned up."""
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "Delete Doc Case"})
    case_id = create.json()["id"]

    with patch("app.api.rag.s3_service.upload_file", new_callable=AsyncMock), \
         patch("app.api.rag.process_document_pipeline.kiq", new_callable=AsyncMock):
        upload = await logged_in_client.post(
            f"/api/v1/rag/cases/{case_id}/documents",
            files={"file": ("test.pdf", io.BytesIO(b"content"), "application/pdf")}
        )
    doc_id = upload.json()["document_id"]

    with patch("app.api.rag.s3_service.delete_file", new_callable=AsyncMock) as mock_s3, \
         patch("app.api.rag.qdrant_service.delete_by_doc_id", new_callable=MagicMock) as mock_qdrant:
        response = await logged_in_client.delete(f"/api/v1/rag/cases/{case_id}/documents/{doc_id}")

    assert response.status_code == 204
    mock_s3.assert_called_once()
    mock_qdrant.assert_called_once()


@pytest.mark.asyncio
async def test_delete_document_not_found(logged_in_client):
    """Test that deleting a non-existent document returns 404."""
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "No Doc Case"})
    case_id = create.json()["id"]
    fake_doc_id = "00000000-0000-0000-0000-000000000000"

    response = await logged_in_client.delete(f"/api/v1/rag/cases/{case_id}/documents/{fake_doc_id}")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_create_chat_success(logged_in_client):
    """Test creating a chat within a case."""
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "Chat Case"})
    case_id = create.json()["id"]

    response = await logged_in_client.post(f"/api/v1/rag/cases/{case_id}/chats", json={
        "title": "My First Chat"
    })
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "My First Chat"
    assert data["case_id"] == case_id


@pytest.mark.asyncio
async def test_create_chat_case_not_found(logged_in_client):
    """Test creating a chat for a non-existent case returns 404."""
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = await logged_in_client.post(f"/api/v1/rag/cases/{fake_id}/chats", json={
        "title": "Ghost Chat"
    })
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_chats(logged_in_client):
    """Test listing chats for a case."""
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "List Chats Case"})
    case_id = create.json()["id"]
    await logged_in_client.post(f"/api/v1/rag/cases/{case_id}/chats", json={"title": "Chat 1"})
    await logged_in_client.post(f"/api/v1/rag/cases/{case_id}/chats", json={"title": "Chat 2"})

    response = await logged_in_client.get(f"/api/v1/rag/cases/{case_id}/chats")
    assert response.status_code == 200
    assert len(response.json()) == 2


@pytest.mark.asyncio
async def test_get_chat_success(logged_in_client):
    """Test retrieving a specific chat."""
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "Get Chat Case"})
    case_id = create.json()["id"]
    chat = await logged_in_client.post(f"/api/v1/rag/cases/{case_id}/chats", json={"title": "Get Me"})
    chat_id = chat.json()["id"]

    response = await logged_in_client.get(f"/api/v1/rag/cases/{case_id}/chats/{chat_id}")
    assert response.status_code == 200
    assert response.json()["id"] == chat_id


@pytest.mark.asyncio
async def test_update_chat_success(logged_in_client):
    """Test updating a chat title."""
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "Update Chat Case"})
    case_id = create.json()["id"]
    chat = await logged_in_client.post(f"/api/v1/rag/cases/{case_id}/chats", json={"title": "Old Chat"})
    chat_id = chat.json()["id"]

    response = await logged_in_client.patch(f"/api/v1/rag/cases/{case_id}/chats/{chat_id}", json={
        "title": "New Chat Title"
    })
    assert response.status_code == 200
    assert response.json()["title"] == "New Chat Title"


@pytest.mark.asyncio
async def test_delete_chat_success(logged_in_client):
    """Test deleting a chat cleans up Qdrant vectors."""
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "Delete Chat Case"})
    case_id = create.json()["id"]
    chat = await logged_in_client.post(f"/api/v1/rag/cases/{case_id}/chats", json={"title": "Delete Me"})
    chat_id = chat.json()["id"]

    with patch("app.api.rag.qdrant_service.delete_by_chat_id", new_callable=MagicMock) as mock_qdrant:
        response = await logged_in_client.delete(f"/api/v1/rag/cases/{case_id}/chats/{chat_id}")

    assert response.status_code == 204
    mock_qdrant.assert_called_once()

    get_response = await logged_in_client.get(f"/api/v1/rag/cases/{case_id}/chats/{chat_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_delete_chat_not_found(logged_in_client):
    """Test deleting a non-existent chat returns 404."""
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "No Chat Case"})
    case_id = create.json()["id"]
    fake_chat_id = "00000000-0000-0000-0000-000000000000"

    response = await logged_in_client.delete(f"/api/v1/rag/cases/{case_id}/chats/{fake_chat_id}")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_send_message_no_context_returns_failure_message(logged_in_client):
    """Test that send_message returns explicit failure when no context is found."""
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "Message Case"})
    case_id = create.json()["id"]
    chat = await logged_in_client.post(f"/api/v1/rag/cases/{case_id}/chats", json={"title": "Message Chat"})
    chat_id = chat.json()["id"]

    mock_retrieval_result = {
        "document_chunks": [],
        "chat_history_chunks": [],
        "has_relevant_context": False,
        "query_type": "new_question",
        "exact_entity": None,
        "doc_hint": None,
        "sub_queries": ["test query"]
    }

    with patch("app.api.rag.retrieval_service.retrieve", new_callable=AsyncMock, return_value=mock_retrieval_result), \
         patch("app.api.rag.retrieval_service.build_context_prompt", return_value=""), \
         patch("app.api.rag.should_push_chat_history", new_callable=AsyncMock, return_value=False):

        response = await logged_in_client.post(
            f"/api/v1/rag/cases/{case_id}/chats/{chat_id}/message",
            json={"content": "What is clause 9b?"}
        )

    assert response.status_code == 200
    full_response = response.text
    assert "could not find" in full_response.lower()


@pytest.mark.asyncio
async def test_send_message_with_context_calls_llm(logged_in_client):
    """Test that send_message calls the LLM when context is found."""
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "LLM Call Case"})
    case_id = create.json()["id"]
    chat = await logged_in_client.post(f"/api/v1/rag/cases/{case_id}/chats", json={"title": "LLM Chat"})
    chat_id = chat.json()["id"]

    mock_retrieval_result = {
        "document_chunks": [{"text": "Clause 9b states...", "doc_name": "contract.pdf", "score": 0.9, "source": "document", "doc_id": "abc"}],
        "chat_history_chunks": [],
        "has_relevant_context": True,
        "query_type": "new_question",
        "exact_entity": "clause 9b",
        "doc_hint": None,
        "sub_queries": ["clause 9b"]
    }

    async def mock_stream(*args, **kwargs):
        import json
        yield f"data: {json.dumps({'type': 'content', 'text': 'Clause 9b states the termination terms.'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    with patch("app.api.rag.retrieval_service.retrieve", new_callable=AsyncMock, return_value=mock_retrieval_result), \
         patch("app.api.rag.retrieval_service.build_context_prompt", return_value="[Source: contract.pdf]\nClause 9b states..."), \
         patch("app.api.rag.should_push_chat_history", new_callable=AsyncMock, return_value=False), \
         patch("app.api.rag.llm_service.stream_response", side_effect=mock_stream):

        response = await logged_in_client.post(
            f"/api/v1/rag/cases/{case_id}/chats/{chat_id}/message",
            json={"content": "What is clause 9b?"}
        )

    assert response.status_code == 200
    assert "Clause 9b" in response.text


@pytest.mark.asyncio
async def test_send_message_chat_not_found(logged_in_client):
    """Test sending a message to a non-existent chat returns 404."""
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "Ghost Chat Case"})
    case_id = create.json()["id"]
    fake_chat_id = "00000000-0000-0000-0000-000000000000"

    response = await logged_in_client.post(
        f"/api/v1/rag/cases/{case_id}/chats/{fake_chat_id}/message",
        json={"content": "Hello?"}
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_chat_messages(logged_in_client):
    """Test retrieving chat message history."""
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "History Case"})
    case_id = create.json()["id"]
    chat = await logged_in_client.post(f"/api/v1/rag/cases/{case_id}/chats", json={"title": "History Chat"})
    chat_id = chat.json()["id"]

    response = await logged_in_client.get(f"/api/v1/rag/cases/{case_id}/chats/{chat_id}/messages")
    assert response.status_code == 200
    data = response.json()
    assert "chat" in data
    assert "messages" in data
    assert isinstance(data["messages"], list)

# --------------------------------------------------------------------------- #
# Deletion must reach the vector store and the object store
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_delete_document_awaits_vector_deletion(logged_in_client):
    """The vector delete is a coroutine and must be awaited, not threaded.

    ``asyncio.to_thread(qdrant_service.delete_by_doc_id, ...)`` only scheduled the
    thread and dropped the coroutine it produced, so the vectors survived the
    deletion while the endpoint returned 204.
    """
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "Await Case"})
    case_id = create.json()["id"]

    with patch("app.api.rag.s3_service.upload_file", new_callable=AsyncMock), \
         patch("app.api.rag.process_document_pipeline.kiq", new_callable=AsyncMock):
        upload = await logged_in_client.post(
            f"/api/v1/rag/cases/{case_id}/documents",
            files={"file": ("test.pdf", io.BytesIO(b"content"), "application/pdf")}
        )
    doc_id = upload.json()["document_id"]

    with patch("app.api.rag.s3_service.delete_file", new_callable=AsyncMock), \
         patch("app.api.rag.qdrant_service.delete_by_doc_id", new_callable=AsyncMock) as mock_qdrant:
        response = await logged_in_client.delete(f"/api/v1/rag/cases/{case_id}/documents/{doc_id}")

    assert response.status_code == 204
    mock_qdrant.assert_awaited_once_with(doc_id)


@pytest.mark.asyncio
async def test_delete_case_purges_vectors_and_stored_objects(logged_in_client):
    """Deleting a case must not leave its contract text searchable."""
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "Purge Case"})
    case_id = create.json()["id"]

    with patch("app.api.rag.s3_service.upload_file", new_callable=AsyncMock), \
         patch("app.api.rag.process_document_pipeline.kiq", new_callable=AsyncMock) as mock_pipeline:
        for name in ("first.pdf", "second.pdf"):
            await logged_in_client.post(
                f"/api/v1/rag/cases/{case_id}/documents",
                files={"file": (name, io.BytesIO(b"content"), "application/pdf")}
            )
    s3_keys = [call.kwargs["s3_key"] for call in mock_pipeline.call_args_list]

    with patch("app.api.rag.s3_service.delete_file", new_callable=AsyncMock) as mock_s3, \
         patch("app.api.rag.qdrant_service.delete_by_case_id", new_callable=AsyncMock) as mock_qdrant:
        response = await logged_in_client.delete(f"/api/v1/rag/cases/{case_id}")

    assert response.status_code == 204
    mock_qdrant.assert_awaited_once_with(case_id)
    deleted_keys = [call.args[0] for call in mock_s3.await_args_list]
    assert sorted(deleted_keys) == sorted(s3_keys)
    assert len(deleted_keys) == 2

    assert (await logged_in_client.get(f"/api/v1/rag/cases/{case_id}")).status_code == 404


@pytest.mark.asyncio
async def test_delete_case_removes_the_real_vectors(logged_in_client, in_memory_qdrant):
    """End-to-end against a real (in-process) Qdrant: the chunks are gone."""
    from qdrant_client import models
    from app.services.qdrant import qdrant_service

    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "Real Purge Case"})
    case_id = create.json()["id"]

    qdrant_service.dense_dim = 4
    await qdrant_service.ensure_collections()
    await qdrant_service.upsert_documents([
        models.PointStruct(
            id="11111111-1111-1111-1111-111111111111",
            vector={"dense": [1.0, 0.0, 0.0, 0.0],
                    "sparse": models.SparseVector(indices=[0], values=[1.0])},
            payload={"text": "indemnity", "doc_name": "acme.pdf", "doc_id": "d1",
                     "case_id": case_id, "user_id": "u1", "source": "document"},
        )
    ])
    assert await qdrant_service.count_points(qdrant_service.documents_collection, case_id) == 1

    with patch("app.api.rag.s3_service.delete_file", new_callable=AsyncMock):
        response = await logged_in_client.delete(f"/api/v1/rag/cases/{case_id}")

    assert response.status_code == 204
    assert await qdrant_service.count_points(qdrant_service.documents_collection, case_id) == 0


@pytest.mark.asyncio
async def test_delete_document_survives_a_storage_failure(logged_in_client):
    """A failing object delete must not leave the row and its vectors behind."""
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "Storage Failure Case"})
    case_id = create.json()["id"]

    with patch("app.api.rag.s3_service.upload_file", new_callable=AsyncMock), \
         patch("app.api.rag.process_document_pipeline.kiq", new_callable=AsyncMock):
        upload = await logged_in_client.post(
            f"/api/v1/rag/cases/{case_id}/documents",
            files={"file": ("test.pdf", io.BytesIO(b"content"), "application/pdf")}
        )
    doc_id = upload.json()["document_id"]

    with patch("app.api.rag.s3_service.delete_file", new_callable=AsyncMock,
               side_effect=RuntimeError("bucket unavailable")), \
         patch("app.api.rag.qdrant_service.delete_by_doc_id", new_callable=AsyncMock) as mock_qdrant:
        response = await logged_in_client.delete(f"/api/v1/rag/cases/{case_id}/documents/{doc_id}")

    assert response.status_code == 204
    mock_qdrant.assert_awaited_once_with(doc_id)
    assert (await logged_in_client.get(
        f"/api/v1/rag/cases/{case_id}/documents/{doc_id}"
    )).status_code == 404


@pytest.mark.asyncio
async def test_delete_chat_awaits_vector_deletion(logged_in_client):
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "Await Chat Case"})
    case_id = create.json()["id"]
    chat = await logged_in_client.post(f"/api/v1/rag/cases/{case_id}/chats", json={"title": "Chat"})
    chat_id = chat.json()["id"]

    with patch("app.api.rag.qdrant_service.delete_by_chat_id", new_callable=AsyncMock) as mock_qdrant:
        response = await logged_in_client.delete(f"/api/v1/rag/cases/{case_id}/chats/{chat_id}")

    assert response.status_code == 204
    mock_qdrant.assert_awaited_once_with(chat_id)


# --------------------------------------------------------------------------- #
# Downloading the original file
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_download_document_returns_a_presigned_url(logged_in_client):
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "Download Case"})
    case_id = create.json()["id"]

    with patch("app.api.rag.s3_service.upload_file", new_callable=AsyncMock), \
         patch("app.api.rag.process_document_pipeline.kiq", new_callable=AsyncMock):
        upload = await logged_in_client.post(
            f"/api/v1/rag/cases/{case_id}/documents",
            files={"file": ("contract.pdf", io.BytesIO(b"content"), "application/pdf")}
        )
    doc_id = upload.json()["document_id"]

    with patch("app.api.rag.s3_service.get_presigned_url", new_callable=AsyncMock,
               return_value="https://minio.local/contract.pdf?sig=abc") as mock_url:
        response = await logged_in_client.get(
            f"/api/v1/rag/cases/{case_id}/documents/{doc_id}/download"
        )

    assert response.status_code == 200
    body = response.json()
    assert body["url"] == "https://minio.local/contract.pdf?sig=abc"
    assert body["filename"] == "contract.pdf"
    assert body["document_id"] == doc_id
    assert body["expires_in"] > 0
    mock_url.assert_awaited_once()


@pytest.mark.asyncio
async def test_download_document_not_found(logged_in_client):
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "No Doc Download Case"})
    case_id = create.json()["id"]

    response = await logged_in_client.get(
        f"/api/v1/rag/cases/{case_id}/documents/00000000-0000-0000-0000-000000000000/download"
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_download_inline_paste_is_rejected(logged_in_client):
    """Pasted text has no object behind it, so there is nothing to sign."""
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "Paste Download Case"})
    case_id = create.json()["id"]
    chat = await logged_in_client.post(f"/api/v1/rag/cases/{case_id}/chats", json={"title": "Chat"})
    chat_id = chat.json()["id"]

    retrieval_result = {
        "document_chunks": [],
        "chat_history_chunks": [],
        "has_relevant_context": False,
        "query_type": "new_question",
        "exact_entity": None,
        "doc_hint": None,
        "sub_queries": ["query"],
    }

    with patch("app.api.rag.retrieval_service.retrieve", new_callable=AsyncMock,
               return_value=retrieval_result), \
         patch("app.api.rag.retrieval_service.build_context_prompt", return_value=""), \
         patch("app.api.rag.should_push_chat_history", new_callable=AsyncMock, return_value=False), \
         patch("app.api.rag.process_inline_paste_pipeline.kiq", new_callable=AsyncMock):
        await logged_in_client.post(
            f"/api/v1/rag/cases/{case_id}/chats/{chat_id}/message",
            json={"content": "Summarise this", "pasted_text": "Clause 1: ..."}
        )

    history = await logged_in_client.get(
        f"/api/v1/rag/cases/{case_id}/chats/{chat_id}/messages"
    )
    attachment_ids = [
        message["attachment_doc_id"]
        for message in history.json()["messages"]
        if message["attachment_doc_id"]
    ]
    assert attachment_ids

    response = await logged_in_client.get(
        f"/api/v1/rag/cases/{case_id}/documents/{attachment_ids[0]}/download"
    )

    assert response.status_code == 400
    assert "pasted" in response.json()["detail"].lower()


# --------------------------------------------------------------------------- #
# Uploads and message ordering
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_upload_without_an_extension_gets_a_safe_suffix(logged_in_client):
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "Extension Case"})
    case_id = create.json()["id"]

    with patch("app.api.rag.s3_service.upload_file", new_callable=AsyncMock) as mock_upload, \
         patch("app.api.rag.process_document_pipeline.kiq", new_callable=AsyncMock):
        response = await logged_in_client.post(
            f"/api/v1/rag/cases/{case_id}/documents",
            files={"file": ("README", io.BytesIO(b"content"), "text/plain")}
        )

    assert response.status_code == 201
    s3_key = mock_upload.await_args.args[1]
    assert s3_key.endswith(".bin")


@pytest.mark.asyncio
async def test_message_sequence_indexes_stay_contiguous(logged_in_client):
    """Four stored messages get indexes 0..3 and the counter agrees."""
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "Sequence Case"})
    case_id = create.json()["id"]
    chat = await logged_in_client.post(f"/api/v1/rag/cases/{case_id}/chats", json={"title": "Chat"})
    chat_id = chat.json()["id"]

    retrieval_result = {
        "document_chunks": [],
        "chat_history_chunks": [],
        "has_relevant_context": False,
        "query_type": "new_question",
        "exact_entity": None,
        "doc_hint": None,
        "sub_queries": ["query"],
    }

    with patch("app.api.rag.retrieval_service.retrieve", new_callable=AsyncMock,
               return_value=retrieval_result), \
         patch("app.api.rag.retrieval_service.build_context_prompt", return_value=""), \
         patch("app.api.rag.should_push_chat_history", new_callable=AsyncMock, return_value=False):
        for question in ("First question?", "Second question?"):
            response = await logged_in_client.post(
                f"/api/v1/rag/cases/{case_id}/chats/{chat_id}/message",
                json={"content": question}
            )
            assert response.status_code == 200

    history = await logged_in_client.get(
        f"/api/v1/rag/cases/{case_id}/chats/{chat_id}/messages"
    )
    body = history.json()
    assert [message["sequence_index"] for message in body["messages"]] == [0, 1, 2, 3]
    assert body["chat"]["message_count"] == 4


@pytest.mark.asyncio
async def test_sequence_stays_contiguous_when_a_stream_dies(logged_in_client):
    """A turn that never produced an answer must not burn a sequence index."""
    create = await logged_in_client.post("/api/v1/rag/cases", json={"title": "Broken Stream Case"})
    case_id = create.json()["id"]
    chat = await logged_in_client.post(f"/api/v1/rag/cases/{case_id}/chats", json={"title": "Chat"})
    chat_id = chat.json()["id"]

    retrieval_result = {
        "document_chunks": [],
        "chat_history_chunks": [],
        "has_relevant_context": False,
        "query_type": "new_question",
        "exact_entity": None,
        "doc_hint": None,
        "sub_queries": ["query"],
    }

    async def empty_stream(*args, **kwargs):
        if False:  # pragma: no cover - an async generator that yields nothing
            yield ""

    with patch("app.api.rag.retrieval_service.retrieve", new_callable=AsyncMock,
               return_value=retrieval_result), \
         patch("app.api.rag.retrieval_service.build_context_prompt", return_value=""), \
         patch("app.api.rag.should_push_chat_history", new_callable=AsyncMock, return_value=False), \
         patch("app.api.rag.llm_service.stream_response", side_effect=empty_stream):
        first = await logged_in_client.post(
            f"/api/v1/rag/cases/{case_id}/chats/{chat_id}/message",
            json={"content": "First question?"}
        )
    assert first.status_code == 200

    with patch("app.api.rag.retrieval_service.retrieve", new_callable=AsyncMock,
               return_value=retrieval_result), \
         patch("app.api.rag.retrieval_service.build_context_prompt", return_value=""), \
         patch("app.api.rag.should_push_chat_history", new_callable=AsyncMock, return_value=False):
        second = await logged_in_client.post(
            f"/api/v1/rag/cases/{case_id}/chats/{chat_id}/message",
            json={"content": "Second question?"}
        )
    assert second.status_code == 200

    history = await logged_in_client.get(
        f"/api/v1/rag/cases/{case_id}/chats/{chat_id}/messages"
    )
    body = history.json()
    # user(0), user(1), assistant(2): the dead turn left no index behind.
    assert [message["sequence_index"] for message in body["messages"]] == [0, 1, 2]
    assert body["chat"]["message_count"] == 3
