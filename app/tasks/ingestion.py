import uuid
from app.tkq import broker
from qdrant_client.models import PointStruct
from app.core.database import AsyncSessionLocal
from app.models.rag import Document, DocumentStatus, ChatMessage, Chat, MessageRole
from app.services.s3 import s3_service
from app.services.unstructured import unstructured_service
from app.services.embeddings import embedding_service
from app.services.sparse import sparse_embedder
from app.services.qdrant import qdrant_service
from sqlalchemy import select
from app.core.logger import log


# Token threshold for pushing chat history to Qdrant
# Estimated as len(text) / 3.5 chars per token
CHAT_HISTORY_TOKEN_THRESHOLD = 6000
CHARS_PER_TOKEN = 3.5


def _estimate_tokens(text: str) -> float:
    return len(text) / CHARS_PER_TOKEN


def _build_points(
    texts: list[str],
    dense_vecs: list[list[float]],
    payload_base: dict
) -> list[PointStruct]:
    sparse_vecs = sparse_embedder.embed_documents(texts)
    points = []
    for text, dense, sparse in zip(texts, dense_vecs, sparse_vecs):
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
                text_content.encode("utf-8"), doc.filename, use_hi_res=False
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
    """
    Push unchecked chat exchange pairs to Qdrant.
    Each point = one user+assistant pair to preserve full Q&A context.
    Triggered when accumulated token estimate of unchecked history exceeds threshold.
    """
    log.info("chat_history_push_started", chat_id=chat_id)

    async with AsyncSessionLocal() as db:
        try:
            # Fetch chat to get user_id and case_id for payload (issue #3)
            chat_result = await db.execute(select(Chat).where(Chat.id == chat_id))
            chat = chat_result.scalar_one_or_none()
            if not chat:
                log.warning("chat_not_found_for_history_push", chat_id=chat_id)
                return

            # Fetch all unchecked non-system messages ordered by sequence
            result = await db.execute(
                select(ChatMessage)
                .where(
                    ChatMessage.chat_id == chat_id,
                    ChatMessage.is_in_qdrant == False,
                    ChatMessage.role != MessageRole.SYSTEM
                )
                .order_by(ChatMessage.sequence_index.asc())
            )
            messages = result.scalars().all()

            if not messages:
                return

            # Issue #4 — group into user+assistant exchange pairs
            pairs: list[tuple[ChatMessage, ChatMessage]] = []
            i = 0
            while i < len(messages) - 1:
                current = messages[i]
                next_msg = messages[i + 1]
                if current.role == MessageRole.USER and next_msg.role == MessageRole.ASSISTANT:
                    pairs.append((current, next_msg))
                    i += 2
                else:
                    i += 1

            if not pairs:
                log.info("no_complete_pairs_to_push", chat_id=chat_id)
                return

            # Build combined text and collect message ids per pair
            pair_texts = []
            pair_message_ids = []
            for user_msg, assistant_msg in pairs:
                combined = f"User: {user_msg.content}\nAssistant: {assistant_msg.content}"
                pair_texts.append(combined)
                pair_message_ids.append([str(user_msg.id), str(assistant_msg.id)])

            dense_vecs = embedding_service.generate_embeddings(pair_texts)
            sparse_vecs = sparse_embedder.embed_documents(pair_texts)

            qdrant_service.ensure_collections()

            # Build points with message_ids in payload (issue #3 + #4)
            points = []
            for text, dense, sparse_vec, msg_ids, (user_msg, _) in zip(
                pair_texts, dense_vecs, sparse_vecs, pair_message_ids, pairs
            ):
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"dense": dense, "sparse": sparse_vec},
                    payload={
                        "user_id": str(chat.user_id),
                        "case_id": str(chat.case_id),
                        "chat_id": chat_id,
                        "doc_name": f"chat_history_{chat_id[:8]}",
                        "source": "chat_history",
                        "message_ids": msg_ids,
                        "sequence_index": user_msg.sequence_index,
                        "text": text,
                    }
                ))

            qdrant_service.upsert_chat_memory(points)

            # Mark all pushed messages as is_in_qdrant=True
            pushed_ids = {msg_id for pair_ids in pair_message_ids for msg_id in pair_ids}
            for msg in messages:
                if str(msg.id) in pushed_ids:
                    msg.is_in_qdrant = True
            await db.commit()

            log.info("chat_history_push_success", chat_id=chat_id, pairs=len(points))

        except Exception:
            log.exception("chat_history_push_failed", chat_id=chat_id)


async def should_push_chat_history(chat_id: str, db) -> bool:
    """
    Issue #5 — token-based threshold check.
    Estimates token count of all unchecked messages.
    Returns True if accumulated tokens exceed threshold.
    """
    result = await db.execute(
        select(ChatMessage)
        .where(
            ChatMessage.chat_id == chat_id,
            ChatMessage.is_in_qdrant == False,
            ChatMessage.role != MessageRole.SYSTEM
        )
    )
    messages = result.scalars().all()

    total_tokens = sum(_estimate_tokens(m.content) for m in messages)
    should_push = total_tokens >= CHAT_HISTORY_TOKEN_THRESHOLD

    log.info(
        "chat_history_token_check",
        chat_id=chat_id,
        estimated_tokens=int(total_tokens),
        threshold=CHAT_HISTORY_TOKEN_THRESHOLD,
        should_push=should_push
    )
    return should_push