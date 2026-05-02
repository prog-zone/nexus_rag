import os
from unstructured_client import UnstructuredClient
from unstructured_client.models import operations, shared
from app.core.config import settings
from app.core.logger import log


class UnstructuredService:
    def __init__(self):
        self.client = UnstructuredClient(
            api_key_auth=settings.UNSTRUCTURED_API_KEY,
        )

    async def partition_file_content(self, content: bytes, filename: str):
        files = shared.Files(
            content=content,
            file_name=filename,
        )

        req = operations.PartitionRequest(
            partition_parameters=shared.PartitionParameters(
                files=files,
                strategy=shared.Strategy.HI_RES,
                chunking_strategy="by_title",
                max_characters=4000,
                combine_under_n_chars=500,
                overlap=400,
            )
        )

        try:
            res = await self.client.general.partition_async(request=req)
            log.info("unstructured_extraction_success", count=len(res.elements or []))
            return res.elements
        except Exception as e:
            log.error("unstructured_extraction_failed", error=str(e))
            raise


unstructured_service = UnstructuredService()