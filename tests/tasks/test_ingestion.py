"""Ingestion helpers: chunk filtering and embedding-count guard."""

import pytest
from qdrant_client import models

from app.tasks.ingestion import _build_points, _extract_texts


def test_extract_texts_drops_blank_and_missing_chunks():
    elements = [
        {"text": "first clause"},
        {"text": "   "},
        {},
        None,
        {"text": None},
        {"text": "second clause"},
    ]

    assert _extract_texts(elements) == ["first clause", "second clause"]


def test_extract_texts_handles_no_elements():
    assert _extract_texts(None) == []
    assert _extract_texts([]) == []


@pytest.mark.asyncio
async def test_build_points_rejects_a_short_embedding_batch(monkeypatch):
    """``zip`` would silently drop chunks; the mismatch has to raise."""
    async def fake_sparse(texts):
        return [models.SparseVector(indices=[0], values=[1.0]) for _ in texts]

    monkeypatch.setattr("app.tasks.ingestion.sparse_embedder.embed_documents", fake_sparse)

    with pytest.raises(ValueError) as excinfo:
        await _build_points(
            texts=["a", "b", "c"],
            dense_vecs=[[1.0], [2.0]],
            payload_base={"case_id": "c1"},
        )

    message = str(excinfo.value)
    assert "3 chunks" in message and "2 dense" in message


@pytest.mark.asyncio
async def test_build_points_rejects_a_short_sparse_batch(monkeypatch):
    async def fake_sparse(texts):
        return [models.SparseVector(indices=[0], values=[1.0])]

    monkeypatch.setattr("app.tasks.ingestion.sparse_embedder.embed_documents", fake_sparse)

    with pytest.raises(ValueError):
        await _build_points(["a", "b"], [[1.0], [2.0]], {"case_id": "c1"})


@pytest.mark.asyncio
async def test_build_points_carries_the_payload(monkeypatch):
    async def fake_sparse(texts):
        return [models.SparseVector(indices=[0], values=[1.0]) for _ in texts]

    monkeypatch.setattr("app.tasks.ingestion.sparse_embedder.embed_documents", fake_sparse)

    points = await _build_points(["a", "b"], [[1.0], [2.0]], {"case_id": "c1", "doc_id": "d1"})

    assert len(points) == 2
    assert points[0].payload == {"case_id": "c1", "doc_id": "d1", "text": "a"}
    assert points[0].id != points[1].id
