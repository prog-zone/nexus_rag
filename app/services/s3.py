import boto3
from botocore.client import Config
from app.core.config import settings
from app.core.logger import log

class S3Service:
    def __init__(self):
        # If endpoint_url is empty, boto3 uses standard AWS S3
        self.s3 = boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT if settings.S3_ENDPOINT else None,
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
            config=Config(signature_version="s3v4"),
            region_name=settings.S3_REGION,
        )

    async def upload_file(self, file_content: bytes, object_name: str):
        try:
            self.s3.put_object(
                Bucket=settings.S3_BUCKET,
                Key=object_name,
                Body=file_content
            )
            log.info("s3_upload_success", bucket=settings.S3_BUCKET, key=object_name)
            return object_name
        except Exception as e:
            log.error("s3_upload_failed", error=str(e))
            raise

    def get_presigned_url(self, object_name: str, expiration=3600):
        """Generate a temporary URL for Unstructured.io to fetch the file."""
        return self.s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': settings.S3_BUCKET, 'Key': object_name},
            ExpiresIn=expiration
        )
    
    async def get_file_content(self, object_name: str) -> bytes:
        """Fetch raw bytes of the file from S3."""
        try:
            response = self.s3.get_object(Bucket=settings.S3_BUCKET, Key=object_name)
            return response['Body'].read()
        except Exception as e:
            log.error("s3_download_failed", key=object_name, error=str(e))
            raise

s3_service = S3Service()