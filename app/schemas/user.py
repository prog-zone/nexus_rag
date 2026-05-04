import re
from uuid import UUID
from datetime import datetime
from typing import Optional, Annotated
from app.models.user import Role
from pydantic import BaseModel, EmailStr, ConfigDict, AfterValidator, Field, field_validator


def validate_password_strength(v: str) -> str:
    if not re.search(r"[A-Z]", v):
        raise ValueError("Password must contain an uppercase letter")
    if not re.search(r"[a-z]", v):
        raise ValueError("Password must contain a lowercase letter")
    if not re.search(r"\d", v):
        raise ValueError("Password must contain a number")
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
        raise ValueError("Password must contain a special character")
    return v

StrongPassword = Annotated[
    str, 
    Field(min_length=8), 
    AfterValidator(validate_password_strength)
]

class UserBaseSchema(BaseModel):
    email: EmailStr

class UserCreateSchema(UserBaseSchema):
    password: StrongPassword
    full_name: str = Field(..., min_length=2, max_length=100)

class UserSchema(UserBaseSchema):
    id: UUID
    role: Role
    is_verified: bool
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class VerifyEmailRequestSchema(BaseModel):
    email: EmailStr
    code: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class VerifyResetCodeRequest(BaseModel):
    email: EmailStr
    code: str

class ResetPasswordRequest(BaseModel):
    reset_token: str
    new_password: StrongPassword

class ProfileBaseSchema(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    summary: Optional[str] = None

class ProfileSchema(ProfileBaseSchema):
    id: UUID
    user_id: UUID
    model_config = ConfigDict(from_attributes=True)

class ProfileCreateSchema(ProfileBaseSchema):
    pass

class ProfileUpdateSchema(ProfileBaseSchema):
    pass

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: StrongPassword

# Schema for full dashboard view (includes all nested data)
class UserFullSchema(UserSchema):
    profile: Optional[ProfileSchema] = None
