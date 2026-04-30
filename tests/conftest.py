import pytest
import pytest_asyncio
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from typing import AsyncGenerator

from app.main import app
from app.core.database import Base, get_db
from app.core.config import settings

# 1. Define the Test Database URL
# We simply append '_test' to your existing database name to keep it isolated.
TEST_DATABASE_URL = f"postgresql+asyncpg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.POSTGRES_DB}_test"

# 2. Create an isolated Async Engine and Session for testing
test_engine = create_async_engine(
    TEST_DATABASE_URL, 
    echo=False,
    poolclass=NullPool
)
TestingSessionLocal = async_sessionmaker(
    bind=test_engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# 3. Setup and Teardown the Database Schema automatically
@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_test_database():
    """
    Creates fresh tables before the test suite runs, 
    and drops them completely after it finishes.
    """
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    yield  # This is where the tests actually execute
    
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await test_engine.dispose()

# 4. Override the FastAPI dependency
async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
    """Hands out test database sessions instead of real ones."""
    async with TestingSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise

# Apply the override to the FastAPI app
app.dependency_overrides[get_db] = override_get_db