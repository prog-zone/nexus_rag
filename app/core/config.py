from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = ""
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = []

    # Database Configuration
    POSTGRES_USER: str = ""
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = ""
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432

    # JWT Configuration
    JWT_SECRET_KEY: str = ""
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Email Configuration
    MAIL_HOST: str = ""
    MAIL_USERNAME: str = ""
    MAIL_PASSWORD: str = ""
    MAIL_FROM: str = ""
    MAIL_PORT: int = 2525

    # Qdrant Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334

    # Voyage AI Configuration
    VOYAGE_API_KEY: str = ""
    VOYAGE_EMBEDDING_MODEL: str = "voyage-law-2"
    VOYAGE_RERANKER_MODEL: str = "rerank-2.5"

    # OpenAI Configuration
    OPENAI_API_KEY: str = ""
    OPENAI_CHAT_MODEL: str = "gpt-4o"

    # Unstructured API Configuration
    UNSTRUCTURED_API_KEY: str = ""

    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379/0"

    # S3 Configuration
    S3_BUCKET: str = ""
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""
    S3_ENDPOINT: str = ""
    S3_REGION: str = ""

    @property
    def database_url(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.POSTGRES_DB}"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()