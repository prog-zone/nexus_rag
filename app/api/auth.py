import jwt
import uuid
from typing import Annotated
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.core.logger import log
from app.core.config import settings
from app.core.database import get_db
from app.core.security import get_password_hash, verify_password, get_otp_hash, verify_otp_hash, create_access_token, create_refresh_token
from app.core.limiter import limiter
from app.api.deps import get_current_user
from app.models.user import User, Profile, UserRefreshToken
from app.schemas.user import UserCreateSchema, UserSchema, UserBaseSchema, VerifyEmailRequestSchema, ForgotPasswordRequest, ResetPasswordRequest
from app.core.email import send_verification_email, generate_otp, send_reset_password_email


router = APIRouter(prefix="/auth", tags=["auth"])

AsyncDbSession = Annotated[AsyncSession, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]
LoginFormData = Annotated[OAuth2PasswordRequestForm, Depends()]
DB_ERROR_MSG = "Database error occurred"

@router.post("/register", response_model=UserSchema)
async def register(user_in: UserCreateSchema, background_tasks: BackgroundTasks, db: AsyncDbSession):
    # Check if user exists
    query = select(User).where(User.email == user_in.email)
    result = await db.execute(query)
    if result.scalar_one_or_none():
        log.warning("registration_failed_user_exists")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User already exists")
    
    raw_otp = generate_otp()

    # Create user
    new_user = User(
        email=user_in.email,
        hashed_password=get_password_hash(user_in.password),
        verification_code=get_otp_hash(raw_otp),
        verification_expire=datetime.now(timezone.utc) + timedelta(minutes=15),
        profile=Profile(full_name=user_in.full_name)
    )
    
    try:
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
    except Exception as e:
        await db.rollback()
        log.error("registration_db_transaction_failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not register user")

    background_tasks.add_task(send_verification_email, user_in.email, raw_otp)
    
    log.info("user_registered_successfully", user_id=str(new_user.id))
    
    return new_user


@router.post("/verify-email")
@limiter.limit("5/minute")
async def verify_email(
    request: Request,
    body: VerifyEmailRequestSchema,
    db: AsyncDbSession
):
    query = select(User).where(User.email == body.email)
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    invalid_exc = HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST, 
        detail="Invalid or expired verification request"
    )

    if not user or user.is_verified:
        log.warning("verify_email_failed_user_invalid")
        raise invalid_exc
        
    if not user.verification_code or not user.verification_expire:
        log.warning("verify_email_failed_no_code_set")
        raise invalid_exc

    if datetime.now(timezone.utc) > user.verification_expire:
        log.warning("verify_email_failed_expired")
        raise invalid_exc

    if not verify_otp_hash(body.code, user.verification_code):
        log.warning("verify_email_failed_wrong_code")
        raise invalid_exc

    # Mark as verified and cleanup
    user.is_verified = True
    user.verification_code = None
    user.verification_expire = None
    
    try:
        await db.commit()
    except Exception as e:
        await db.rollback()
        log.error("db_transaction_failed", error=str(e), path=request.url.path)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=DB_ERROR_MSG)
    
    log.info("email_verified_successfully", user_id=str(user.id))
    
    return {"message": "Email successfully verified"}


@router.post("/resend-verification")
@limiter.limit("3/minute")
async def resend_verification(
    request: Request,
    body: UserBaseSchema,
    background_tasks: BackgroundTasks,
    db: AsyncDbSession
):
    query = select(User).where(User.email == body.email)
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    if user and not user.is_verified:
        raw_otp = generate_otp()
        user.verification_code = get_otp_hash(raw_otp)
        user.verification_expire = datetime.now(timezone.utc) + timedelta(minutes=15)

        try:
            await db.commit()
            background_tasks.add_task(send_verification_email, user.email, raw_otp)
            log.info("verification_email_resent", user_id=str(user.id))
        except Exception as e:
            await db.rollback()
            log.error("db_transaction_failed", error=str(e), path=request.url.path)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=DB_ERROR_MSG)
    else:
        if not user:
            log.warning("resend_verification_ignored_user_not_found")
        else:
            log.info("resend_verification_ignored_already_verified", user_id=str(user.id))

    return {"message": "If that email exists and is unverified, a new code has been sent."}


@router.post("/login")
@limiter.limit("5/minute")
async def login(
    request: Request,
    response: Response,
    db: AsyncDbSession, 
    form_data: LoginFormData
):
    query = select(User).where(User.email == form_data.username)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        log.warning("login_failed_invalid_credentials")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect email or password")
    
    if not user.is_verified:
        log.warning("login_failed_unverified_email", user_id=str(user.id))
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Please verify your email address to log in")
    
    access_token = create_access_token(user.id)
    refresh_token_str, jti = create_refresh_token(user.id)
    
    try:
        db_refresh_token = UserRefreshToken(
            user_id=user.id,
            token_jti=jti,
            expires_at=datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS) 
        )
        db.add(db_refresh_token)
        await db.commit()
    except Exception as e:
        await db.rollback()
        log.error("login_db_transaction_failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not complete login")

    # Set both tokens as HttpOnly cookies
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token_str,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
    )
    
    log.info("user_logged_in_successfully", user_id=str(user.id))
    
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "message": "Successfully logged in"
    }


@router.post("/refresh")
@limiter.limit("3/minute")
async def refresh_token(request: Request, response: Response, db: AsyncDbSession):
    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token missing")
        
    try:
        payload = jwt.decode(refresh_token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        if payload.get("type") != "refresh":
             log.warning("refresh_failed_invalid_type", path=request.url.path)
             raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")
             
        user_id_str = payload.get("sub")
        jti = payload.get("jti")
        user_id = uuid.UUID(user_id_str)
    except jwt.PyJWTError as e:
        log.warning("refresh_token_decode_failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate refresh token")

    query = select(UserRefreshToken).where(
        UserRefreshToken.user_id == user_id, 
        UserRefreshToken.token_jti == jti
    ).with_for_update()

    result = await db.execute(query)
    db_token = result.scalar_one_or_none()

    if not db_token:
        log.warning("token_theft_detected_wiping_sessions", user_id=str(user_id))
        wipe_query = delete(UserRefreshToken).where(UserRefreshToken.user_id == user_id)
        await db.execute(wipe_query)
        await db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session compromised. Please log in again.")
    
    new_access_token = create_access_token(user_id)
    new_refresh_token, new_jti = create_refresh_token(user_id)

    try:
        await db.delete(db_token)
        
        db_new_token = UserRefreshToken(
            user_id=user_id,
            token_jti=new_jti,
            expires_at=datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        )
        db.add(db_new_token)
        await db.commit()

    except Exception as e:
        await db.rollback()
        log.error("database_transaction_failed_during_refresh", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not refresh session")

    # Set new cookies
    response.set_cookie(
        key="access_token",
        value=new_access_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )
    response.set_cookie(
        key="refresh_token",
        value=new_refresh_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
    )

    log.info("tokens_refreshed_successfully", user_id=str(user_id))

    return {"message": "Tokens refreshed successfully"}


@router.post("/logout")
async def logout(request: Request, response: Response, db: AsyncDbSession):
    refresh_token = request.cookies.get("refresh_token")
    
    if refresh_token:
        try:
            payload = jwt.decode(
                refresh_token, 
                settings.JWT_SECRET_KEY, 
                algorithms=[settings.JWT_ALGORITHM],
                options={"verify_exp": False} 
            )
            jti = payload.get("jti")
            if jti:
                query = delete(UserRefreshToken).where(UserRefreshToken.token_jti == jti)
                await db.execute(query)
                await db.commit()
        except jwt.PyJWTError as e:
            log.warning("logout_token_decode_failed", error=str(e))

    response.delete_cookie(key="access_token", httponly=True, secure=True, samesite="lax")
    response.delete_cookie(key="refresh_token", httponly=True, secure=True, samesite="lax")

    log.info("user_logged_out_successfully")

    return {"message": "Successfully logged out"}


@router.post("/logout-all")
async def logout_all_devices(
    response: Response, 
    db: AsyncDbSession,
    current_user: CurrentUser
):
    query = delete(UserRefreshToken).where(UserRefreshToken.user_id == current_user.id)
    await db.execute(query)
    await db.commit()

    response.delete_cookie(key="access_token", httponly=True, secure=True, samesite="lax")
    response.delete_cookie(key="refresh_token", httponly=True, secure=True, samesite="lax")

    log.info("user_logged_out_all_devices", user_id=str(current_user.id))

    return {
        "message": "Successfully logged out from all devices."
    }


@router.post("/forgot-password")
@limiter.limit("3/minute")
async def forgot_password(
    request: Request,
    body: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: AsyncDbSession
):
    query = select(User).where(User.email == body.email)
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    if user:
        # Generate new OTP, hash it, and set expiration
        raw_otp = generate_otp()
        user.resetpass_code = get_password_hash(raw_otp)
        user.resetpass_expire = datetime.now(timezone.utc) + timedelta(minutes=15)

        try:
            await db.commit()
            background_tasks.add_task(send_reset_password_email, user.email, raw_otp)
            log.info("password_reset_requested", user_id=str(user.id))
        except Exception as e:
            await db.rollback()
            log.error("db_transaction_failed", error=str(e), path=request.url.path)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=DB_ERROR_MSG)
    else:
        log.warning("forgot_password_ignored_user_not_found")

    return {"message": "If that email exists in our system, a password reset code has been sent."}


@router.post("/reset-password")
@limiter.limit("5/minute")
async def reset_password(
    request: Request,
    body: ResetPasswordRequest,
    db: AsyncDbSession
):
    query = select(User).where(User.email == body.email)
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    invalid_exc = HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST, 
        detail="Invalid or expired reset request"
    )

    if not user:
        log.warning("password_reset_failed_user_not_found")
        raise invalid_exc
        
    if not user.resetpass_code or not user.resetpass_expire:
        log.warning("password_reset_failed_no_request_active")
        raise invalid_exc

    if datetime.now(timezone.utc) > user.resetpass_expire:
        log.warning("password_reset_failed_expired")
        raise invalid_exc

    if not verify_password(body.code, user.resetpass_code):
        log.warning("password_reset_failed_wrong_code")
        raise invalid_exc

    user.hashed_password = get_password_hash(body.new_password)
    user.resetpass_code = None
    user.resetpass_expire = None
    
    # SECURITY CRITICAL: Invalidate all existing sessions (Logout from all devices)
    try:
        delete_sessions_query = delete(UserRefreshToken).where(UserRefreshToken.user_id == user.id)
        await db.execute(delete_sessions_query)
        await db.commit()
    except Exception as e:
        await db.rollback()
        log.error("db_transaction_failed", error=str(e), path=request.url.path)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=DB_ERROR_MSG)
    
    log.info("password_reset_successful", user_id=str(user.id))
    
    return {"message": "Password has been reset successfully. All active sessions have been logged out."}