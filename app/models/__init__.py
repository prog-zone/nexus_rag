from app.core.database import Base
from app.models.user import User, Profile, UserRefreshToken

# This makes importing 'Base' in env.py pick up all attached models
__all__ = ["Base", "User", "Profile", "UserRefreshToken"]