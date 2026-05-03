import aioboto3
from botocore.client import Config
from app.core.config import settings
from app.core.logger import log

class S3Service:
    def __init__(self):
        self.session = aioboto3.Session()
        
        self.client_kwargs = {
            "service_name": "s3",
            "endpoint_url": settings.S3_ENDPOINT if settings.S3_ENDPOINT else None,
            "aws_access_key_id": settings.S3_ACCESS_KEY,
            "aws_secret_access_key": settings.S3_SECRET_KEY,
            "config": Config(signature_version="s3v4"),
            "region_name": settings.S3_REGION,
        }

    async def upload_file(self, file_content: bytes, object_name: str) -> str:
        try:
            async with self.session.client(**self.client_kwargs) as s3: # type: ignore
                await s3.put_object(
                    Bucket=settings.S3_BUCKET,
                    Key=object_name,
                    Body=file_content
                )
            log.info("s3_upload_success", bucket=settings.S3_BUCKET, key=object_name)
            return object_name
        except Exception as e:
            log.error("s3_upload_failed", error=str(e))
            raise

    async def get_presigned_url(self, object_name: str, expiration=3600) -> str:
        try:
            async with self.session.client(**self.client_kwargs) as s3: # type: ignore
                return await s3.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': settings.S3_BUCKET, 'Key': object_name},
                    ExpiresIn=expiration
                )
        except Exception as e:
            log.error("s3_presigned_url_failed", key=object_name, error=str(e))
            raise
    
    async def get_file_content(self, object_name: str) -> bytes:
        try:
            async with self.session.client(**self.client_kwargs) as s3: # type: ignore
                response = await s3.get_object(Bucket=settings.S3_BUCKET, Key=object_name)
                return await response['Body'].read()
        except Exception as e:
            log.error("s3_download_failed", key=object_name, error=str(e))
            raise

    async def delete_file(self, object_name: str) -> None:
        try:
            async with self.session.client(**self.client_kwargs) as s3:  # type: ignore
                await s3.delete_object(Bucket=settings.S3_BUCKET, Key=object_name)
            log.info("s3_delete_success", key=object_name)
        except Exception as e:
            log.error("s3_delete_failed", key=object_name, error=str(e))
            raise

s3_service = S3Service()