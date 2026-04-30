from typing import List
from fastapi import Depends, HTTPException, status
from app.models.user import User, Role
from app.api.deps import get_current_user

class RequireRole:
    def __init__(self, allowed_roles: List[Role]):
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have enough privileges to access this resource"
            )
        return current_user

is_admin = RequireRole([Role.ADMIN, Role.SUPERUSER])
is_superuser = RequireRole([Role.SUPERUSER])