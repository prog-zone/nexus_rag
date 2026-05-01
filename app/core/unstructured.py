import os
import asyncio
from unstructured_client import UnstructuredClient
from unstructured_client.models import operations, shared
from app.core.config import settings

client = UnstructuredClient(
    api_key_auth=settings.UNSTRUCTURED_API_KEY,
)

async def process_document(file_path: str):
    with open(file_path, "rb") as f:
        file_content = f.read()

    files = shared.Files(
        content=file_content,
        file_name=os.path.basename(file_path)
    )
    
    req = operations.PartitionRequest(
        partition_parameters=shared.PartitionParameters(
            files=files,
            chunking_strategy="by_title", 
            split_pdf_page=True 
        )
    )
    
    try:
        res = await client.general.partition_async(request=req)
        return res.elements
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None