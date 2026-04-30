import pytest
import jwt
from datetime import datetime, timedelta, timezone
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.core.config import settings

@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="https://testserver")

@pytest.mark.asyncio
async def test_forged_jwt_signature_rejected(client):
    """Test that a JWT signed with the wrong secret key is rejected."""
    # 1. A hacker creates a token with a fake secret key
    payload = {
        "sub": "fake-user-id", 
        "type": "access", 
        "exp": datetime.now(timezone.utc) + timedelta(minutes=15)
    }
    
    hacker_token = jwt.encode(payload, "hacker_secret_key_12345_that_is_long_enough", algorithm="HS256")    
    
    # 2. They inject it into the cookies and try to access a protected route
    client.cookies.set("access_token", hacker_token)
    response = await client.get("/api/v1/user/me")
    
    # 3. The API must catch the fake signature and throw a 401
    assert response.status_code == 401
    assert response.json()["detail"] == "Could not validate credentials"

@pytest.mark.asyncio
async def test_expired_jwt_rejected(client):
    """Test that an old, expired token is completely rejected."""
    # 1. We create a real token using your actual secret key, BUT we backdate it 
    # so it expired 10 minutes ago.
    payload = {
        "sub": "valid-user-id", 
        "type": "access", 
        "exp": datetime.now(timezone.utc) - timedelta(minutes=10) # <-- Expired!
    }
    expired_token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    
    # 2. Try to use the expired token
    client.cookies.set("access_token", expired_token)
    response = await client.get("/api/v1/user/me")
    
    # 3. The API must realize the time has passed and reject it
    assert response.status_code == 401
    assert response.json()["detail"] == "Could not validate credentials"


@pytest.mark.asyncio
async def test_cors_allowed_origin(client):
    """Test that requests from an allowed frontend origin get CORS headers."""
    # Simulate a "preflight" request from your actual frontend
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST"
    }
    response = await client.options("/api/v1/auth/login", headers=headers)
    
    # If CORS is configured correctly, FastAPI will say "Yes, this origin is allowed"
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] in ["http://localhost:3000", "*"]


@pytest.mark.asyncio
async def test_cors_blocked_origin(client):
    """Test that requests from malicious origins do NOT get CORS approval."""
    # Simulate a request from a hacker's custom frontend
    headers = {"Origin": "https://evil-hacker-site.com"}
    response = await client.get("/api/v1/user/me", headers=headers)
    
    # The API might process the request, but it will STRIP the CORS headers.
    # This means the hacker's web browser will physically block them from reading the response!
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.asyncio
async def test_security_headers_present(client):
    """Test that all anti-clickjacking and strict HTTPS headers are attached."""
    # We can hit any endpoint for this, even a health check
    response = await client.get("/api/v1/user/me")
    
    headers = response.headers
    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert headers.get("X-Frame-Options") == "DENY"
    assert "Strict-Transport-Security" in headers