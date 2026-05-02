import jwt
import uuid
import structlog
from typing import List
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.database import get_db
from app.models.user import User, Role
from app.core.logger import log

# This looks for the "Authorization: Bearer <token>" header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db), 
    token_header: str | None = Depends(oauth2_scheme)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token = request.cookies.get("access_token") or token_header

    if not token:
         raise credentials_exception

    try:
        payload = jwt.decode(
            token, 
            settings.JWT_SECRET_KEY, 
            algorithms=[settings.JWT_ALGORITHM]
        )
        if payload.get("type") == "refresh":
            log.warning("refresh_token_used_for_access", path=request.url.path)
            raise credentials_exception
            
        user_id_str = payload.get("sub")
        if user_id_str is None:
            raise credentials_exception
            
        user_id = uuid.UUID(user_id_str)
        
    except jwt.ExpiredSignatureError:
        raise credentials_exception
    except (jwt.PyJWTError, ValueError) as e:
        log.warning("token_decode_failed", path=request.url.path, error=str(e))
        raise credentials_exception

    # 2. Fetch the user from the DB
    query = select(User).where(User.id == user_id)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    
    if user is None:
        log.error("auth_user_not_found_in_db", user_id=str(user_id))
        raise credentials_exception
    
    structlog.contextvars.bind_contextvars(user_id=str(user.id))
    
    return user