from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.core.logger import log
from app.core.database import get_db
from app.core.limiter import limiter
from app.api.deps import get_current_user
from app.core.security import get_password_hash, verify_password
from app.models.user import User, Profile, UserRefreshToken
from app.schemas.user import UserFullSchema, ProfileSchema, ProfileUpdateSchema, ChangePasswordRequest


router = APIRouter(prefix="/user", tags=["user"])
AsyncDbSession = Annotated[AsyncSession, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]
DB_ERROR_MSG = "Database error occurred"


"""Retrieve the authenticated user's full profile."""
@router.get("/me", response_model=UserFullSchema)
async def get_user_profile(
    request: Request,
    current_user: CurrentUser, 
    db: AsyncDbSession
    ):
    try:
        await db.refresh(current_user, ["profile"])
        return current_user
    except Exception as e:
        log.error("get_profile_failed", user_id=str(current_user.id), error=str(e), path=request.url.path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Could not retrieve profile"
        )


"""Update specific fields in the user's profile."""
@router.patch("/me", response_model=ProfileSchema)
async def update_profile(
    request: Request,
    profile_in: ProfileUpdateSchema,
    current_user: CurrentUser,
    db: AsyncDbSession
):
    try:
        # 1. Fetch the profile
        query = select(Profile).where(Profile.user_id == current_user.id)
        result = await db.execute(query)
        profile = result.scalar_one_or_none()

        if not profile:
            log.warning("profile_not_found_creating_new", user_id=str(current_user.id))
            profile = Profile(**profile_in.model_dump(), user_id=current_user.id)
            db.add(profile)
        else:
            # 2. Update fields
            update_data = profile_in.model_dump(exclude_unset=True)
            for key, value in update_data.items():
                setattr(profile, key, value)

        # 3. Commit and refresh
        await db.commit()
        await db.refresh(profile)
        
        log.info("profile_updated_successfully", user_id=str(current_user.id))
        return profile

    except Exception as e:
        await db.rollback()
        log.error("profile_update_db_failed", user_id=str(current_user.id), error=str(e), path=request.url.path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Database error occurred while updating profile"
        )


@router.post("/change-password")
@limiter.limit("5/minute")
async def change_password(
    request: Request,
    response: Response,
    body: ChangePasswordRequest,
    current_user: CurrentUser,
    db: AsyncDbSession
):
    if not await verify_password(body.current_password, current_user.hashed_password):
        log.warning("password_change_failed_wrong_current_password", user_id=str(current_user.id))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Incorrect current password"
        )

    if body.current_password == body.new_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="New password cannot be the same as the current password"
        )

    current_user.hashed_password = await get_password_hash(body.new_password)
    
    # SECURITY CRITICAL: Log out all devices (including this one) 
    # to force them to log back in with the new credentials.
    try:
        delete_sessions_query = delete(UserRefreshToken).where(UserRefreshToken.user_id == current_user.id)
        await db.execute(delete_sessions_query)
        await db.commit()
    except Exception as e:
        await db.rollback()
        log.error("db_transaction_failed", error=str(e), path=request.url.path)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=DB_ERROR_MSG)
    
    # Clear the cookies for the current session
    response.delete_cookie(key="access_token", httponly=True, secure=True, samesite="lax")
    response.delete_cookie(key="refresh_token", httponly=True, secure=True, samesite="lax")
    
    log.info("password_changed_successfully", user_id=str(current_user.id))
    
    return {"message": "Password changed successfully. Please log in with your new password."}


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_account(
    request: Request,
    response: Response,
    current_user: CurrentUser,
    db: AsyncDbSession
):
    """
    Permanently delete the user account and all associated data.
    """
    try:
        await db.delete(current_user)
        await db.commit()

        # Clear authentication cookies
        response.delete_cookie(key="access_token", httponly=True, secure=True, samesite="lax")
        response.delete_cookie(key="refresh_token", httponly=True, secure=True, samesite="lax")

        log.info("user_account_deleted", user_id=str(current_user.id))
        return None

    except Exception as e:
        await db.rollback()
        log.error("user_deletion_failed", user_id=str(current_user.id), error=str(e), path=request.url.path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while deleting your account"
        )