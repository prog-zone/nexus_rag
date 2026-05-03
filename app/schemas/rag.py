import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field
from app.models.rag import CaseStatus, DocumentStatus, DocumentSource, MessageRole


class CreateCaseSchema(BaseModel):
    title: str = Field(..., min_length=2, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)

class UpdateCaseSchema(BaseModel):
    title: Optional[str] = Field(None, min_length=2, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    status: Optional[CaseStatus] = None

class CaseSchema(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    title: str
    description: Optional[str]
    status: CaseStatus
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


class DocumentSchema(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    case_id: uuid.UUID
    chat_id: Optional[uuid.UUID]
    filename: str
    source: DocumentSource
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)

class DocumentStatusSchema(BaseModel):
    document_id: uuid.UUID
    status: DocumentStatus
    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_document(cls, doc) -> "DocumentStatusSchema":
        return cls(document_id=doc.id, status=doc.status)


class CreateChatSchema(BaseModel):
    title: str = Field(default="New Chat", min_length=1, max_length=200)

class UpdateChatSchema(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)

class ChatSchema(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    case_id: uuid.UUID
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


class SendMessageSchema(BaseModel):
    content: str = Field(..., min_length=1, max_length=50000)
    pasted_text: Optional[str] = Field(None, max_length=100000)

class ChatMessageSchema(BaseModel):
    id: uuid.UUID
    chat_id: uuid.UUID
    role: MessageRole
    content: str
    sequence_index: int
    has_attachment: bool
    attachment_doc_id: Optional[uuid.UUID]
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class ChatHistorySchema(BaseModel):
    chat: ChatSchema
    messages: list[ChatMessageSchema]