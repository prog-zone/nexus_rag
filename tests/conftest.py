"""Test bootstrap.

The suite runs against a throwaway SQLite database by default, so `pytest` works
on a clean checkout with no PostgreSQL server running. Point ``TEST_DATABASE_URL``
at any async SQLAlchemy URL (for example
``postgresql+asyncpg://user:password@localhost:5432/project_db_test``) to run the
same suite against PostgreSQL.

Nothing here touches the application database from ``settings.database_url``:
the test engine is created separately and injected through ``get_db``.
"""

import os
import pathlib
import random
import tempfile

# Configuration the app needs but a test run should not depend on a developer's
# .env file. Must be set before app.core.config is imported below.
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("BACKEND_CORS_ORIGINS", '["http://localhost:3000"]')

import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from qdrant_client import AsyncQdrantClient  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from typing import AsyncGenerator  # noqa: E402

from app.main import app  # noqa: E402
from app.core.database import Base, get_db  # noqa: E402
from app.services.qdrant import qdrant_service  # noqa: E402


def _default_sqlite_url() -> tuple[str, pathlib.Path]:
    """A per-process SQLite file, so the suite can run without a DB server."""
    path = pathlib.Path(tempfile.gettempdir()) / f"nexus_rag_test_{os.getpid()}.db"
    return f"sqlite+aiosqlite:///{path}", path


_explicit_test_url = os.environ.get("TEST_DATABASE_URL")
if _explicit_test_url:
    TEST_DATABASE_URL = _explicit_test_url
    _sqlite_path: pathlib.Path | None = None
else:
    TEST_DATABASE_URL, _sqlite_path = _default_sqlite_url()

# Isolated async engine/session for tests.
test_engine = create_async_engine(TEST_DATABASE_URL, echo=False, poolclass=NullPool)
TestingSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_test_database():
    """Create fresh tables before the suite and drop them afterwards."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    yield  # This is where the tests actually execute

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await test_engine.dispose()

    if _sqlite_path is not None:
        _sqlite_path.unlink(missing_ok=True)


@pytest_asyncio.fixture(autouse=True)
async def in_memory_qdrant(monkeypatch):
    """Give every test an isolated in-process Qdrant instead of localhost:6333."""
    client = AsyncQdrantClient(location=":memory:")
    monkeypatch.setattr(qdrant_service, "client", client)
    yield client
    await client.close()


@pytest_asyncio.fixture(scope="session")
async def logged_in_client(session_mocker):
    """An authenticated client backed by the test database."""
    session_mocker.patch("app.api.auth.send_verification_email")
    session_mocker.patch("app.api.auth.generate_otp", return_value="123456")

    random_ip = f"10.0.0.{random.randint(1, 250)}"
    transport = ASGITransport(app=app, client=(random_ip, 12345))

    async with AsyncClient(transport=transport, base_url="https://testserver") as client:
        await client.post("/api/v1/auth/register", json={
            "email": "raguser@example.com",
            "password": "StrongPassword123!",
            "full_name": "RAG Test User"
        })
        await client.post("/api/v1/auth/verify-email", json={
            "email": "raguser@example.com",
            "code": "123456"
        })
        await client.post("/api/v1/auth/login", data={
            "username": "raguser@example.com",
            "password": "StrongPassword123!"
        })
        yield client


async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
    """Hands out test database sessions instead of real ones."""
    async with TestingSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


app.dependency_overrides[get_db] = override_get_db
