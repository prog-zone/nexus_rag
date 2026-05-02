import uuid
from app.tkq import broker
from qdrant_client.models import PointStruct, SparseVector
from app.core.database import AsyncSessionLocal
from app.models.rag import Document, DocumentStatus, ChatMessage, Chat, MessageRole
from app.services.s3 import s3_service
from app.services.unstructured import unstructured_service
from app.services.embeddings import embedding_service
from app.services.qdrant import qdrant_service
from sqlalchemy import select
from app.core.logger import log


def _build_bm25_sparse_vector(text: str) -> SparseVector:
    terms = text.lower().split()
    term_freq: dict[int, float] = {}
    for term in terms:
        term_id = hash(term) % (2 ** 31)
        term_freq[term_id] = term_freq.get(term_id, 0) + 1.0

    return SparseVector(
        indices=list(term_freq.keys()),
        values=list(term_freq.values())
    )


def _build_points(
    texts: list[str],
    dense_vecs: list[list[float]],
    payload_base: dict
) -> list[PointStruct]:
    points = []
    for text, dense in zip(texts, dense_vecs):
        sparse = _build_bm25_sparse_vector(text)
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dense,
                "sparse": sparse
            },
            payload={**payload_base, "text": text}
        ))
    return points


@broker.task
async def process_document_pipeline(doc_id: str, s3_key: str):
    log.info("ingestion_task_started", doc_id=doc_id)

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if not doc:
            return

        doc.status = DocumentStatus.PROCESSING
        await db.commit()

        try:
            content = await s3_service.get_file_content(s3_key)
            elements = await unstructured_service.partition_file_content(content, doc.filename) or []

            texts = [el["text"] for el in elements if "text" in el]

            if not texts:
                log.warning("no_content_extracted", doc_id=doc_id)
                doc.status = DocumentStatus.FAILED
                return

            dense_vecs = embedding_service.generate_embeddings(texts)

            qdrant_service.ensure_collections()
            points = _build_points(
                texts, dense_vecs,
                payload_base={
                    "user_id": str(doc.user_id),
                    "case_id": str(doc.case_id),
                    "doc_id": doc_id,
                    "doc_name": doc.filename,
                    "source": "document",
                }
            )

            qdrant_service.upsert_documents(points)
            doc.status = DocumentStatus.COMPLETED
            log.info("ingestion_task_success", doc_id=doc_id, chunks=len(points))

        except Exception:
            log.exception("ingestion_task_failed", doc_id=doc_id)
            doc.status = DocumentStatus.FAILED
        finally:
            await db.commit()


@broker.task
async def process_inline_paste_pipeline(doc_id: str, text_content: str):
    log.info("inline_paste_ingestion_started", doc_id=doc_id)

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if not doc:
            return

        doc.status = DocumentStatus.PROCESSING
        await db.commit()

        try:
            elements = await unstructured_service.partition_file_content(
                text_content.encode("utf-8"), doc.filename
            ) or []

            texts = [el["text"] for el in elements if "text" in el]

            if not texts:
                log.warning("no_content_extracted_from_paste", doc_id=doc_id)
                doc.status = DocumentStatus.FAILED
                return

            dense_vecs = embedding_service.generate_embeddings(texts)

            qdrant_service.ensure_collections()
            points = _build_points(
                texts, dense_vecs,
                payload_base={
                    "user_id": str(doc.user_id),
                    "case_id": str(doc.case_id),
                    "chat_id": str(doc.chat_id),
                    "doc_id": doc_id,
                    "doc_name": doc.filename,
                    "source": "inline_paste",
                }
            )

            qdrant_service.upsert_chat_memory(points)
            doc.status = DocumentStatus.COMPLETED
            log.info("inline_paste_ingestion_success", doc_id=doc_id, chunks=len(points))

        except Exception:
            log.exception("inline_paste_ingestion_failed", doc_id=doc_id)
            doc.status = DocumentStatus.FAILED
        finally:
            await db.commit()

@broker.task
async def push_chat_history_to_qdrant(chat_id: str):
    log.info("chat_history_push_started", chat_id=chat_id)

    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(
                select(ChatMessage)
                .where(
                    ChatMessage.chat_id == chat_id,
                    ChatMessage.is_in_qdrant == False
                )
            )
            messages = result.scalars().all()

            if not messages:
                return

            texts = [m.content for m in messages]
            dense_vecs = embedding_service.generate_embeddings(texts)

            qdrant_service.ensure_collections()
            points = _build_points(
                texts, dense_vecs,
                payload_base={
                    "chat_id": chat_id,
                    "source": "chat_history",
                }
            )

            qdrant_service.upsert_chat_memory(points)

            for msg in messages:
                msg.is_in_qdrant = True
            await db.commit()

            log.info("chat_history_push_success", chat_id=chat_id, chunks=len(points))

        except Exception:
            log.exception("chat_history_push_failed", chat_id=chat_id)