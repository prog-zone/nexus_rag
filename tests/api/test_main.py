import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.core.config import settings

@pytest.mark.asyncio
async def test_health_check():
    """Test that the API boots up and the health endpoint responds correctly."""
    
    # ASGITransport is the modern way to test FastAPI apps directly 
    # without needing to spin up a real live server.
    transport = ASGITransport(app=app)
    
    async with AsyncClient(transport=transport, base_url="https://testserver") as ac:
        response = await ac.get("/api/v1/health")
    
    # Assert we get a 200 OK
    assert response.status_code == 200
    
    # Assert the payload matches exactly what main.py returns
    assert response.json() == {"message": f"{settings.PROJECT_NAME} API is running"}