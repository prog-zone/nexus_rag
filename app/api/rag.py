import uuid
from typing import Annotated
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, UploadFile, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_current_user
from app.core.database import get_db
from app.core.logger import log
from app.services.s3 import s3_service
from app.services.llm import llm_service
from app.services.retrieval import retrieval_service
from app.models.user import User
from app.models.rag import Case, Chat, ChatMessage, Document, DocumentStatus, DocumentSource, CaseStatus, MessageRole
from app.schemas.rag import (
    CreateCaseSchema, UpdateCaseSchema, CaseSchema,
    DocumentSchema, DocumentStatusSchema,
    CreateChatSchema, UpdateChatSchema, ChatSchema,
    SendMessageSchema, ChatMessageSchema, ChatHistorySchema
)
from app.tasks.ingestion import process_document_pipeline, process_inline_paste_pipeline

router = APIRouter(prefix="/rag", tags=["rag"])

AsyncDbSession = Annotated[AsyncSession, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]


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
    result = await db.execute(
        select(Case).where(Case.id == case_id, Case.user_id == current_user.id)
    )
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")
    return case


@router.patch("/cases/{case_id}", response_model=CaseSchema)
async def update_case(
    case_id: uuid.UUID,
    body: UpdateCaseSchema,
    current_user: CurrentUser,
    db: AsyncDbSession
):
    result = await db.execute(
        select(Case).where(Case.id == case_id, Case.user_id == current_user.id)
    )
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

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
    result = await db.execute(
        select(Case).where(Case.id == case_id, Case.user_id == current_user.id)
    )
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")
    await db.delete(case)
    await db.commit()
    log.info("case_deleted", case_id=str(case_id), user_id=str(current_user.id))


@router.post("/cases/{case_id}/documents", response_model=DocumentStatusSchema, status_code=status.HTTP_201_CREATED)
async def upload_document(
    case_id: uuid.UUID,
    file: UploadFile,
    current_user: CurrentUser,
    db: AsyncDbSession
):
    result = await db.execute(
        select(Case).where(Case.id == case_id, Case.user_id == current_user.id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

    filename = file.filename or "unknown"
    file_ext = filename.split(".")[-1]
    s3_key = f"uploads/{current_user.id}/{case_id}/{uuid.uuid4()}.{file_ext}"

    content = await file.read()
    await s3_service.upload_file(content, s3_key)

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
    result = await db.execute(
        select(Case).where(Case.id == case_id, Case.user_id == current_user.id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

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
    result = await db.execute(
        select(Document).where(
            Document.id == doc_id,
            Document.case_id == case_id,
            Document.user_id == current_user.id
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc


@router.delete("/cases/{case_id}/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    case_id: uuid.UUID,
    doc_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncDbSession
):
    result = await db.execute(
        select(Document).where(
            Document.id == doc_id,
            Document.case_id == case_id,
            Document.user_id == current_user.id
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
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
    result = await db.execute(
        select(Case).where(Case.id == case_id, Case.user_id == current_user.id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

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
    result = await db.execute(
        select(Case).where(Case.id == case_id, Case.user_id == current_user.id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

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
    result = await db.execute(
        select(Chat).where(
            Chat.id == chat_id,
            Chat.case_id == case_id,
            Chat.user_id == current_user.id
        )
    )
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    return chat


@router.patch("/cases/{case_id}/chats/{chat_id}", response_model=ChatSchema)
async def update_chat(
    case_id: uuid.UUID,
    chat_id: uuid.UUID,
    body: UpdateChatSchema,
    current_user: CurrentUser,
    db: AsyncDbSession
):
    result = await db.execute(
        select(Chat).where(
            Chat.id == chat_id,
            Chat.case_id == case_id,
            Chat.user_id == current_user.id
        )
    )
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

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
    result = await db.execute(
        select(Chat).where(
            Chat.id == chat_id,
            Chat.case_id == case_id,
            Chat.user_id == current_user.id
        )
    )
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
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
    result = await db.execute(
        select(Chat).where(
            Chat.id == chat_id,
            Chat.case_id == case_id,
            Chat.user_id == current_user.id
        )
    )
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.chat_id == chat_id)
        .order_by(ChatMessage.sequence_index.asc())
    )
    messages = result.scalars().all()
    return {"chat": chat, "messages": messages}


INLINE_PASTE_THRESHOLD = 500

@router.post("/cases/{case_id}/chats/{chat_id}/message")
async def send_message(
    case_id: uuid.UUID,
    chat_id: uuid.UUID,
    body: SendMessageSchema,
    current_user: CurrentUser,
    db: AsyncDbSession
):
    result = await db.execute(
        select(Chat).where(
            Chat.id == chat_id,
            Chat.case_id == case_id,
            Chat.user_id == current_user.id
        )
    )
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

    user_message = ChatMessage(
        chat_id=chat_id,
        role=MessageRole.USER,
        content=body.content,
        sequence_index=chat.message_count,
        is_in_qdrant=False,
        has_attachment=False
    )
    db.add(user_message)
    await db.commit()
    await db.refresh(user_message)

    if len(body.content) > INLINE_PASTE_THRESHOLD:
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
            text_content=body.content
        )

    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.chat_id == chat_id)
        .order_by(ChatMessage.sequence_index.desc())
        .limit(10)
    )
    recent_messages = list(reversed(result.scalars().all()))

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
        has_context=retrieval_result["has_relevant_context"]
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