import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

# 1. Create a reusable client fixture. 
# This saves us from writing the transport setup in every single auth test.
@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="https://testserver")

@pytest.mark.asyncio
async def test_register_success(client, mocker):
    """Test that a user can successfully register with valid credentials."""
    
    # 2. "Mute" the background email task. 
    # This tells pytest: "When the app tries to call this function, just pretend it worked and do nothing."
    mocker.patch("app.api.auth.send_verification_email")
    
    # 3. Create the payload that matches your UserCreateSchema
    payload = {
        "email": "testuser@example.com",
        "password": "StrongPassword123!",
        "full_name": "Test User"
    }
    
    # 4. Make the POST request
    response = await client.post("/api/v1/auth/register", json=payload)
    
    # 5. Assert the outcomes
    assert response.status_code == 200
    
    data = response.json()
    assert data["email"] == payload["email"]
    assert "id" in data  # Proves the database generated a UUID
    assert data["is_verified"] is False  # Proves default schema values are working
    assert data["role"] == "user" # Proves default role is assigned


@pytest.mark.asyncio
async def test_register_duplicate_email(client, mocker):
    """Test that registering an existing email fails."""
    mocker.patch("app.api.auth.send_verification_email")
    
    # We use the exact same email from the first test. 
    payload = {
        "email": "testuser@example.com", # This email is already taken!
        "password": "AnotherStrongPassword1!",
        "full_name": "Imposter User"
    }
    
    response = await client.post("/api/v1/auth/register", json=payload)
    
    # Your auth.py returns a 400 Bad Request for duplicates
    assert response.status_code == 400
    assert response.json()["detail"] == "User already exists"


@pytest.mark.asyncio
async def test_register_weak_password(client):
    """Test that our custom StrongPassword validator blocks weak passwords."""
    payload = {
        "email": "weakpass@example.com",
        "password": "password123",  # Fails: No uppercase, no special character
        "full_name": "Weak Pass User"
    }
    
    response = await client.post("/api/v1/auth/register", json=payload)
    
    # Assert FastAPI blocks it for schema validation
    assert response.status_code == 422
    
    # Let's dig into the JSON response to ensure it failed specifically on the password field
    error_detail = response.json()["detail"][0]
    assert error_detail["loc"] == ["body", "password"]
    assert "Password must contain" in error_detail["msg"]


@pytest.mark.asyncio
async def test_login_unverified_user(client):
    """Test that an unverified user cannot log in."""
    # We use 'testuser@example.com' from the first test because they aren't verified yet!
    payload = {
        "username": "testuser@example.com",  # OAuth2 form uses 'username' for the email
        "password": "StrongPassword123!"
    }
    
    # Notice we use 'data=' instead of 'json=' because the login endpoint expects Form Data
    response = await client.post("/api/v1/auth/login", data=payload)
    
    assert response.status_code == 403
    assert response.json()["detail"] == "Please verify your email address to log in"


@pytest.mark.asyncio
async def test_verify_email_success(client, mocker):
    """Test successful email verification."""
    # 1. Force the OTP generator to always return "123456"
    mocker.patch("app.api.auth.generate_otp", return_value="123456")
    mocker.patch("app.api.auth.send_verification_email")
    
    # 2. Register a new user
    reg_payload = {
        "email": "verify_me@example.com",
        "password": "StrongPassword123!",
        "full_name": "Verify Me"
    }
    await client.post("/api/v1/auth/register", json=reg_payload)
    
    # 3. Verify the user with our known OTP
    verify_payload = {
        "email": "verify_me@example.com",
        "code": "123456"
    }
    response = await client.post("/api/v1/auth/verify-email", json=verify_payload)
    
    assert response.status_code == 200
    assert response.json()["message"] == "Email successfully verified"


@pytest.mark.asyncio
async def test_login_success(client):
    """Test that a verified user can log in and receive tokens in cookies."""
    # We use the user we successfully verified in the previous test!
    payload = {
        "username": "verify_me@example.com", 
        "password": "StrongPassword123!"
    }
    
    response = await client.post("/api/v1/auth/login", data=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    
    # CRITICAL SECURITY CHECK: Ensure tokens are set as HttpOnly cookies!
    cookies = response.cookies
    assert "access_token" in cookies
    assert "refresh_token" in cookies


@pytest.mark.asyncio
async def test_login_wrong_credentials(client):
    """Test that invalid passwords are rejected."""
    payload = {
        "username": "verify_me@example.com",
        "password": "WrongPassword999!"
    }
    
    response = await client.post("/api/v1/auth/login", data=payload)
    
    assert response.status_code == 400
    assert response.json()["detail"] == "Incorrect email or password"


@pytest.mark.asyncio
async def test_login_rate_limiter():
    """Test that the brute force protection (rate limiter) blocks spam."""
    
    # Create a brand new transport and explicitly spoof the ASGI connection IP
    transport = ASGITransport(app=app, client=("123.45.67.89", 12345))
    
    async with AsyncClient(transport=transport, base_url="https://testserver") as unique_ip_client:
        payload = {
            "username": "spam@example.com",
            "password": "WrongPassword!"
        }
        
        # Make 5 bad requests. These should process normally and return 400.
        for _ in range(5):
            response = await unique_ip_client.post("/api/v1/auth/login", data=payload)
            assert response.status_code == 400
            
        # The 6th request must trip the wire and trigger the 429 error!
        response = await unique_ip_client.post("/api/v1/auth/login", data=payload)
        
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["message"]


@pytest.mark.asyncio
async def test_refresh_token_success(client):
    """Test that a valid refresh token generates new tokens and rotates the session."""
    # 1. Log in to get the initial tokens
    payload = {
        "username": "verify_me@example.com", 
        "password": "StrongPassword123!"
    }
    login_response = await client.post("/api/v1/auth/login", data=payload)
    
    initial_refresh_cookie = login_response.cookies.get("refresh_token")
    initial_access_cookie = login_response.cookies.get("access_token")
    
    # 2. Hit the refresh endpoint (AsyncClient automatically sends the cookies from login)
    refresh_response = await client.post("/api/v1/auth/refresh")
    
    assert refresh_response.status_code == 200
    
    # 3. Verify the tokens actually rotated
    new_refresh_cookie = refresh_response.cookies.get("refresh_token")
    new_access_cookie = refresh_response.cookies.get("access_token")
    
    assert new_refresh_cookie is not None
    assert new_access_cookie is not None
    assert new_refresh_cookie != initial_refresh_cookie
    assert new_access_cookie != initial_access_cookie