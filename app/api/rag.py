import uuid
import pathlib
from typing import Annotated
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, UploadFile, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.api.deps import get_current_user
from app.core.database import get_db
from app.core.logger import log
from app.core.config import settings
from app.services.s3 import s3_service
from app.services.qdrant import qdrant_service
from app.services.llm import llm_service
from app.services.retrieval import retrieval_service
from app.models.user import User
from app.models.rag import Case, Chat, ChatMessage, Document, DocumentStatus, DocumentSource, CaseStatus, MessageRole
from app.schemas.rag import (
    CreateCaseSchema, UpdateCaseSchema, CaseSchema,
    DocumentSchema, DocumentStatusSchema, DocumentDownloadSchema,
    CreateChatSchema, UpdateChatSchema, ChatSchema,
    SendMessageSchema, ChatHistorySchema
)
from app.tasks.ingestion import process_document_pipeline, process_inline_paste_pipeline, push_chat_history_to_qdrant, should_push_chat_history

router = APIRouter(prefix="/rag", tags=["rag"])

AsyncDbSession = Annotated[AsyncSession, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]

RECENT_PAIRS_MIN = 5
UNKNOWN_FILE_EXTENSION = "bin"


async def _get_case_or_404(case_id: uuid.UUID, user_id: uuid.UUID, db: AsyncSession) -> Case:
    result = await db.execute(
        select(Case).where(Case.id == case_id, Case.user_id == user_id)
    )
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")
    return case


async def _get_chat_or_404(
    case_id: uuid.UUID, chat_id: uuid.UUID, user_id: uuid.UUID, db: AsyncSession
) -> Chat:
    result = await db.execute(
        select(Chat).where(
            Chat.id == chat_id,
            Chat.case_id == case_id,
            Chat.user_id == user_id
        )
    )
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    return chat


async def _get_document_or_404(
    case_id: uuid.UUID, doc_id: uuid.UUID, user_id: uuid.UUID, db: AsyncSession
) -> Document:
    result = await db.execute(
        select(Document).where(
            Document.id == doc_id,
            Document.case_id == case_id,
            Document.user_id == user_id
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc


def _stored_objects(documents: list[Document]) -> list[str]:
    """S3 keys of the documents that actually have an object behind them."""
    return [
        doc.s3_key
        for doc in documents
        if doc.source == DocumentSource.UPLOAD and doc.s3_key
    ]


async def _purge_document_assets(doc: Document) -> None:
    """Remove the bytes and the vectors of a document, best effort.

    Failures are logged instead of raised: a single unreachable storage or vector
    backend must not stop the caller from finishing the rest of the deletion.
    """
    if doc.source == DocumentSource.UPLOAD and doc.s3_key:
        try:
            await s3_service.delete_file(doc.s3_key)
        except Exception:
            log.exception("document_object_purge_failed", doc_id=str(doc.id), s3_key=doc.s3_key)

    try:
        await qdrant_service.delete_by_doc_id(str(doc.id))
    except Exception:
        log.exception("document_vectors_purge_failed", doc_id=str(doc.id))


async def _purge_case_assets(case_id: uuid.UUID, documents: list[Document]) -> None:
    """Remove every stored object and every vector that belongs to a case."""
    for s3_key in _stored_objects(documents):
        try:
            await s3_service.delete_file(s3_key)
        except Exception:
            log.exception("case_object_purge_failed", case_id=str(case_id), s3_key=s3_key)

    try:
        await qdrant_service.delete_by_case_id(str(case_id))
    except Exception:
        log.exception("case_vectors_purge_failed", case_id=str(case_id))


async def _get_recent_messages(chat_id: uuid.UUID, db: AsyncSession) -> list[ChatMessage]:
    unchecked_result = await db.execute(
        select(ChatMessage)
        .where(
            ChatMessage.chat_id == chat_id,
            ChatMessage.is_in_qdrant == False,
            ChatMessage.role != MessageRole.SYSTEM
        )
        .order_by(ChatMessage.sequence_index.asc())
    )
    unchecked = unchecked_result.scalars().all()

    unchecked_pairs = sum(
        1 for i in range(len(unchecked) - 1)
        if unchecked[i].role == MessageRole.USER and unchecked[i + 1].role == MessageRole.ASSISTANT
    )

    if unchecked_pairs >= RECENT_PAIRS_MIN:
        return list(unchecked)

    checked_result = await db.execute(
        select(ChatMessage)
        .where(
            ChatMessage.chat_id == chat_id,
            ChatMessage.is_in_qdrant == True,
            ChatMessage.role != MessageRole.SYSTEM
        )
        .order_by(ChatMessage.sequence_index.desc())
        .limit(RECENT_PAIRS_MIN * 2)
    )
    checked = list(reversed(checked_result.scalars().all()))

    log.info(
        "recent_messages_topped_up",
        chat_id=str(chat_id),
        unchecked_pairs=unchecked_pairs,
        checked_added=len(checked)
    )

    return checked + list(unchecked)


@router.post("/cases", response_model=CaseSchema, status_code=status.HTTP_201_CREATED)
async def create_case(body: CreateCaseSchema, current_user: CurrentUser, db: AsyncDbSession):
    new_case = Case(
        user_id=current_user.id,
        title=body.title,
        description=body.description,
        status=CaseStatus.ACTIVE
    )
    db.add(new_case)
    await db.commit()
    await db.refresh(new_case)
    log.info("case_created", case_id=str(new_case.id), user_id=str(current_user.id))
    return new_case


@router.get("/cases", response_model=list[CaseSchema])
async def list_cases(current_user: CurrentUser, db: AsyncDbSession):
    result = await db.execute(
        select(Case)
        .where(Case.user_id == current_user.id)
        .order_by(Case.created_at.desc())
    )
    return result.scalars().all()


@router.get("/cases/{case_id}", response_model=CaseSchema)
async def get_case(case_id: uuid.UUID, current_user: CurrentUser, db: AsyncDbSession):
    return await _get_case_or_404(case_id, current_user.id, db)


@router.patch("/cases/{case_id}", response_model=CaseSchema)
async def update_case(
    case_id: uuid.UUID,
    body: UpdateCaseSchema,
    current_user: CurrentUser,
    db: AsyncDbSession
):
    case = await _get_case_or_404(case_id, current_user.id, db)

    if body.title is not None:
        case.title = body.title
    if body.description is not None:
        case.description = body.description
    if body.status is not None:
        case.status = body.status

    await db.commit()
    await db.refresh(case)
    log.info("case_updated", case_id=str(case_id), user_id=str(current_user.id))
    return case


@router.delete("/cases/{case_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_case(case_id: uuid.UUID, current_user: CurrentUser, db: AsyncDbSession):
    case = await _get_case_or_404(case_id, current_user.id, db)

    documents_result = await db.execute(
        select(Document).where(Document.case_id == case_id)
    )
    documents = list(documents_result.scalars().all())

    # Purging the stored objects and the vectors before the row disappears: the
    # cascade only reaches PostgreSQL, so deleting the case alone left every
    # uploaded contract and its chunks behind, still retrievable by queries that
    # only filter on case_id/user_id.
    await _purge_case_assets(case_id, documents)

    await db.delete(case)
    await db.commit()
    log.info(
        "case_deleted",
        case_id=str(case_id),
        user_id=str(current_user.id),
        documents=len(documents)
    )


@router.post("/cases/{case_id}/documents", response_model=DocumentStatusSchema, status_code=status.HTTP_201_CREATED)
async def upload_document(
    case_id: uuid.UUID,
    file: UploadFile,
    current_user: CurrentUser,
    db: AsyncDbSession
):
    await _get_case_or_404(case_id, current_user.id, db)

    doc_count_result = await db.execute(
        select(func.count()).where(
            Document.case_id == case_id,
            Document.source == DocumentSource.UPLOAD
        )
    )
    if (doc_count_result.scalar() or 0) >= settings.MAX_DOCS_PER_CASE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Case has reached the maximum limit of {settings.MAX_DOCS_PER_CASE} documents."
        )

    if file.size and file.size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds the maximum allowed size of {settings.MAX_FILE_SIZE_MB}MB."
        )

    filename = file.filename or "unknown"
    file_ext = pathlib.PurePath(filename).suffix.lstrip(".").lower() or UNKNOWN_FILE_EXTENSION
    s3_key = f"uploads/{current_user.id}/{case_id}/{uuid.uuid4()}.{file_ext}"

    await s3_service.upload_file(file, s3_key)

    new_doc = Document(
        user_id=current_user.id,
        case_id=case_id,
        chat_id=None,
        filename=filename,
        s3_key=s3_key,
        source=DocumentSource.UPLOAD,
        status=DocumentStatus.PENDING
    )
    db.add(new_doc)
    await db.commit()
    await db.refresh(new_doc)

    await process_document_pipeline.kiq(doc_id=str(new_doc.id), s3_key=s3_key)
    log.info("document_uploaded", doc_id=str(new_doc.id), case_id=str(case_id))
    return DocumentStatusSchema.from_document(new_doc)


@router.get("/cases/{case_id}/documents", response_model=list[DocumentSchema])
async def list_documents(case_id: uuid.UUID, current_user: CurrentUser, db: AsyncDbSession):
    await _get_case_or_404(case_id, current_user.id, db)

    result = await db.execute(
        select(Document).where(
            Document.case_id == case_id,
            Document.source == DocumentSource.UPLOAD
        ).order_by(Document.created_at.desc())
    )
    return result.scalars().all()


@router.get("/cases/{case_id}/documents/{doc_id}", response_model=DocumentSchema)
async def get_document(
    case_id: uuid.UUID,
    doc_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncDbSession
):
    return await _get_document_or_404(case_id, doc_id, current_user.id, db)


@router.get("/cases/{case_id}/documents/{doc_id}/download", response_model=DocumentDownloadSchema)
async def download_document(
    case_id: uuid.UUID,
    doc_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncDbSession
):
    """Short-lived pre-signed URL for the original file.

    Uploads were write-only from the API's point of view: the lawyer could add a
    contract and never fetch it back through the product.
    """
    doc = await _get_document_or_404(case_id, doc_id, current_user.id, db)

    if doc.source != DocumentSource.UPLOAD or not doc.s3_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This document was pasted into a chat and has no stored file."
        )

    url = await s3_service.get_presigned_url(doc.s3_key, expiration=settings.DOCUMENT_URL_EXPIRE_SECONDS)
    log.info("document_download_url_issued", doc_id=str(doc_id), case_id=str(case_id))
    return DocumentDownloadSchema(
        document_id=doc.id,
        filename=doc.filename,
        url=url,
        expires_in=settings.DOCUMENT_URL_EXPIRE_SECONDS
    )


@router.delete("/cases/{case_id}/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    case_id: uuid.UUID,
    doc_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncDbSession
):
    doc = await _get_document_or_404(case_id, doc_id, current_user.id, db)

    # ``delete_by_doc_id`` is a coroutine: handing it to ``asyncio.to_thread``
    # returned a coroutine object that nothing awaited, so the vectors were never
    # deleted and the "deleted" document kept answering later questions.
    await _purge_document_assets(doc)

    await db.delete(doc)
    await db.commit()
    log.info("document_deleted", doc_id=str(doc_id), case_id=str(case_id))


@router.post("/cases/{case_id}/chats", response_model=ChatSchema, status_code=status.HTTP_201_CREATED)
async def create_chat(
    case_id: uuid.UUID,
    body: CreateChatSchema,
    current_user: CurrentUser,
    db: AsyncDbSession
):
    await _get_case_or_404(case_id, current_user.id, db)

    new_chat = Chat(
        user_id=current_user.id,
        case_id=case_id,
        title=body.title,
        message_count=0
    )
    db.add(new_chat)
    await db.commit()
    await db.refresh(new_chat)
    log.info("chat_created", chat_id=str(new_chat.id), case_id=str(case_id))
    return new_chat


@router.get("/cases/{case_id}/chats", response_model=list[ChatSchema])
async def list_chats(case_id: uuid.UUID, current_user: CurrentUser, db: AsyncDbSession):
    await _get_case_or_404(case_id, current_user.id, db)

    result = await db.execute(
        select(Chat)
        .where(Chat.case_id == case_id)
        .order_by(Chat.created_at.desc())
    )
    return result.scalars().all()


@router.get("/cases/{case_id}/chats/{chat_id}", response_model=ChatSchema)
async def get_chat(
    case_id: uuid.UUID,
    chat_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncDbSession
):
    return await _get_chat_or_404(case_id, chat_id, current_user.id, db)


@router.patch("/cases/{case_id}/chats/{chat_id}", response_model=ChatSchema)
async def update_chat(
    case_id: uuid.UUID,
    chat_id: uuid.UUID,
    body: UpdateChatSchema,
    current_user: CurrentUser,
    db: AsyncDbSession
):
    chat = await _get_chat_or_404(case_id, chat_id, current_user.id, db)

    chat.title = body.title
    await db.commit()
    await db.refresh(chat)
    log.info("chat_updated", chat_id=str(chat_id))
    return chat


@router.delete("/cases/{case_id}/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(
    case_id: uuid.UUID,
    chat_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncDbSession
):
    chat = await _get_chat_or_404(case_id, chat_id, current_user.id, db)

    try:
        await qdrant_service.delete_by_chat_id(str(chat_id))
    except Exception:
        log.exception("chat_vectors_purge_failed", chat_id=str(chat_id))

    await db.delete(chat)
    await db.commit()
    log.info("chat_deleted", chat_id=str(chat_id), case_id=str(case_id))


@router.get("/cases/{case_id}/chats/{chat_id}/messages", response_model=ChatHistorySchema)
async def get_chat_messages(
    case_id: uuid.UUID,
    chat_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncDbSession
):
    chat = await _get_chat_or_404(case_id, chat_id, current_user.id, db)

    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.chat_id == chat_id)
        .order_by(ChatMessage.sequence_index.asc())
    )
    messages = result.scalars().all()
    return {"chat": chat, "messages": messages}


@router.post("/cases/{case_id}/chats/{chat_id}/message")
async def send_message(
    case_id: uuid.UUID,
    chat_id: uuid.UUID,
    body: SendMessageSchema,
    current_user: CurrentUser,
    db: AsyncDbSession
):
    chat = await _get_chat_or_404(case_id, chat_id, current_user.id, db)

    user_message = ChatMessage(
        chat_id=chat_id,
        role=MessageRole.USER,
        content=body.content,
        sequence_index=chat.message_count,
        is_in_qdrant=False,
        has_attachment=False
    )
    db.add(user_message)
    # Keep the counter in step with the rows that exist, rather than with the
    # number of turns that were *supposed* to be written.
    chat.message_count = user_message.sequence_index + 1
    await db.commit()
    await db.refresh(user_message)

    if body.pasted_text:
        paste_doc = Document(
            user_id=current_user.id,
            case_id=case_id,
            chat_id=chat_id,
            filename=f"pasted_text_{user_message.id}",
            s3_key="",
            source=DocumentSource.INLINE_PASTE,
            status=DocumentStatus.PENDING
        )
        db.add(paste_doc)
        await db.commit()
        await db.refresh(paste_doc)

        user_message.has_attachment = True
        user_message.attachment_doc_id = paste_doc.id
        await db.commit()

        await process_inline_paste_pipeline.kiq(
            doc_id=str(paste_doc.id),
            text_content=body.pasted_text
        )

    if await should_push_chat_history(chat_id=str(chat_id), db=db):
        await push_chat_history_to_qdrant.kiq(chat_id=str(chat_id))

    recent_messages = await _get_recent_messages(chat_id=chat_id, db=db)

    retrieval_result = await retrieval_service.retrieve(
        query=body.content,
        case_id=str(case_id),
        chat_id=str(chat_id),
        user_id=str(current_user.id),
        recent_messages=[
            {"role": m.role.value, "content": m.content}
            for m in recent_messages
        ]
    )
    context_prompt = retrieval_service.build_context_prompt(retrieval_result)

    log.info(
        "message_pipeline_complete",
        chat_id=str(chat_id),
        has_context=retrieval_result["has_relevant_context"],
        recent_messages_count=len(recent_messages)
    )

    return StreamingResponse(
        llm_service.stream_response(
            query=body.content,
            context_prompt=context_prompt,
            recent_messages=recent_messages,
            has_relevant_context=retrieval_result["has_relevant_context"],
            chat=chat,
            user_message=user_message,
            db=db
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
