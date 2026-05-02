import uuid
from app.tkq import broker
from qdrant_client.models import PointStruct
from app.core.database import AsyncSessionLocal
from app.models.document import Document, DocumentStatus
from app.services.s3 import s3_service
from app.services.unstructured import unstructured_service
from app.services.embeddings import embedding_service
from app.services.qdrant import qdrant_service
from sqlalchemy import select
from app.core.logger import log

@broker.task
async def process_document_pipeline(doc_id: str, s3_key: str):
    log.info("ingestion_task_started", doc_id=doc_id)
    
    async with AsyncSessionLocal() as db:
        query = select(Document).where(Document.id == doc_id)
        result = await db.execute(query)
        doc = result.scalar_one_or_none()
        if not doc: return

        doc.status = DocumentStatus.PROCESSING
        await db.commit()

        try:
            # 1. Extraction
            content = await s3_service.get_file_content(s3_key)
            elements = await unstructured_service.partition_file_content(content, doc.filename) or []
            
            texts = [el["text"] for el in elements if "text" in el]
            metadatas = [el.get("metadata", {}) for el in elements if "text" in el]

            if not texts:
                log.warning("no_content_extracted", doc_id=doc_id)
                doc.status = DocumentStatus.FAILED
                return

            # 2. Hybrid Embedding Generation
            dense_vecs, sparse_vecs = embedding_service.generate_hybrid_embeddings(texts)

            # 3. Qdrant Preparation
            qdrant_service.ensure_collection()
            points = []
            for i, (text, dense, sparse) in enumerate(zip(texts, dense_vecs, sparse_vecs)):
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector={    # type: ignore
                        "dense": dense.tolist(),
                        "sparse": sparse.as_object()
                    },
                    payload={
                        "doc_id": doc_id,
                        "text": text,
                        "metadata": metadatas[i]
                    }
                ))

            # 4. Final Upsert[cite: 1]
            await qdrant_service.upsert_points(points)
            doc.status = DocumentStatus.COMPLETED
            log.info("ingestion_task_success", doc_id=doc_id, chunks=len(points))

        except Exception as e:
            log.exception("ingestion_task_failed", doc_id=doc_id)
            doc.status = DocumentStatus.FAILED
        finally:
            await db.commit()