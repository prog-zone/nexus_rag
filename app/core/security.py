import jwt
import uuid
import hashlib
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Union
from pwdlib import PasswordHash
from app.core.config import settings

# Uses argon2-cffi
password_hash = PasswordHash.recommended()

async def get_password_hash(password: str) -> str:
    return await asyncio.to_thread(password_hash.hash, password)

async def verify_password(plain_password: str, hashed_password: str) -> bool:
    return await asyncio.to_thread(password_hash.verify, plain_password, hashed_password)

def get_otp_hash(otp: str) -> str:
    return hashlib.sha256(otp.encode()).hexdigest()

def verify_otp_hash(plain_otp: str, hashed_otp: str) -> bool:
    return hashlib.sha256(plain_otp.encode()).hexdigest() == hashed_otp

def create_access_token(subject: Union[str, Any]) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    to_encode = {
        "exp": expire,
        "sub": str(subject),
        "type": "access",
        "jti": str(uuid.uuid4())
        }
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(subject: Union[str, Any]) -> tuple[str, str]:
    jti = str(uuid.uuid4())
    expire = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode = {
        "exp": expire, 
        "sub": str(subject), 
        "type": "refresh",
        "jti": jti
    }
    token = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return token, jti