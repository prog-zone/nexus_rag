import random
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.fixture
def client():
    random_ip = f"192.168.1.{random.randint(1, 250)}"
    transport = ASGITransport(app=app, client=(random_ip, 12345))
    return AsyncClient(transport=transport, base_url="https://testserver")

@pytest.mark.asyncio
async def test_get_me_unauthorized(client):
    """Test that a user without a token is blocked from protected routes."""
    # Try to access the profile directly without logging in
    response = await client.get("/api/v1/user/me")
    
    # 401 Unauthorized is the correct HTTP status for missing/invalid tokens
    assert response.status_code == 401
    assert response.json()["detail"] == "Could not validate credentials"

@pytest.mark.asyncio
async def test_get_me_success(client):
    """Test that an authenticated user can access their protected profile."""
    # 1. Log in to get the secure cookies attached to our client
    login_payload = {
        "username": "verify_me@example.com",
        "password": "StrongPassword123!"
    }
    await client.post("/api/v1/auth/login", data=login_payload)
    
    # 2. Now access the protected route!
    response = await client.get("/api/v1/user/me")
    
    assert response.status_code == 200
    
    # 3. Verify it returned the correct user data
    data = response.json()
    assert data["email"] == "verify_me@example.com"
    assert "id" in data
    assert "profile" in data # Proves your relationship mapping works!


@pytest.mark.asyncio
async def test_update_profile_success(client):
    """Test that a user can update their profile information."""
    # 1. First, we need to log in to get the secure cookie
    login_payload = {
        "username": "verify_me@example.com",
        "password": "StrongPassword123!"
    }
    await client.post("/api/v1/auth/login", data=login_payload)
    
    # 2. Send the PATCH request with some new profile data
    update_payload = {
        "location": "Helsinki, Finland",
        "github": "https://github.com/prog-zone",
        "summary": "FastAPI Developer looking for remote work!"
    }
    
    response = await client.patch("/api/v1/user/me", json=update_payload)
    
    # 3. Assert the response
    assert response.status_code == 200
    data = response.json()
    assert data["location"] == "Helsinki, Finland"
    assert data["github"] == "https://github.com/prog-zone"


@pytest.mark.asyncio
async def test_delete_user_account(client):
    """Test that a user can permanently delete their own account."""
    login_payload = {
        "username": "verify_me@example.com",
        "password": "StrongPassword123!"
    }
    await client.post("/api/v1/auth/login", data=login_payload)
    
    # 1. Delete the account
    response = await client.delete("/api/v1/user/me")
    
    assert response.status_code == 204 
    
    # 2. Verify the account is actually gone
    check_response = await client.get("/api/v1/user/me")
    assert check_response.status_code == 401


@pytest.mark.asyncio
async def test_change_password_success(client, mocker):
    """Test that a user can change their password and old credentials are invalidated."""
    
    # 1. SETUP: Create and log in a fresh user for this test
    mocker.patch("app.api.auth.generate_otp", return_value="999999")
    mocker.patch("app.api.auth.send_verification_email")
    
    await client.post("/api/v1/auth/register", json={
        "email": "changepass@example.com",
        "password": "OldPassword123!",
        "full_name": "Password Changer"
    })
    await client.post("/api/v1/auth/verify-email", json={
        "email": "changepass@example.com",
        "code": "999999"
    })
    await client.post("/api/v1/auth/login", data={
        "username": "changepass@example.com",
        "password": "OldPassword123!"
    })
    
    # 2. EXECUTE: Change the password
    change_payload = {
        "current_password": "OldPassword123!",
        "new_password": "NewStrongPassword456!"
    }
    response = await client.post("/api/v1/user/change-password", json=change_payload)
    
    assert response.status_code == 200
    assert "Password changed successfully" in response.json()["message"]
    
    # 3. VERIFY: Ensure the old cookies were wiped by the server
    assert response.cookies.get("access_token") in [None, '""', ""]
    
    # 4. VERIFY: Trying to log in with the OLD password must fail
    bad_login = await client.post("/api/v1/auth/login", data={
        "username": "changepass@example.com",
        "password": "OldPassword123!"
    })
    assert bad_login.status_code == 400
    
    # 5. VERIFY: Trying to log in with the NEW password must succeed
    good_login = await client.post("/api/v1/auth/login", data={
        "username": "changepass@example.com",
        "password": "NewStrongPassword456!"
    })
    assert good_login.status_code == 200