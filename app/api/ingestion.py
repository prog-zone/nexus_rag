from fastapi import APIRouter, UploadFile, Depends, HTTPException, status
from typing import Annotated
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

from app.api.deps import get_current_user
from app.core.database import get_db
from app.services.s3 import s3_service
from app.models.document import Document, DocumentStatus
from app.tasks.ingestion import process_document_pipeline
from app.models.user import User

router = APIRouter(prefix="/ingestion", tags=["ingestion"])

@router.post("/upload")
async def upload_document(
    file: UploadFile,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
):
    # 1. Generate unique S3 Key
    filename = file.filename or "unknown"
    file_ext = filename.split(".")[-1]
    s3_key = f"uploads/{current_user.id}/{uuid.uuid4()}.{file_ext}"
    
    # 2. Upload to S3
    content = await file.read()
    await s3_service.upload_file(content, s3_key)
    
    # 3. Create DB Record
    new_doc = Document(
        user_id=current_user.id,
        filename=file.filename,
        s3_key=s3_key,
        status=DocumentStatus.PENDING
    )
    db.add(new_doc)
    await db.commit()
    await db.refresh(new_doc)
    
    # 4. Trigger Taskiq Task
    await process_document_pipeline.kiq(doc_id=str(new_doc.id), s3_key=s3_key)
    
    return {"document_id": new_doc.id, "status": "processing"}