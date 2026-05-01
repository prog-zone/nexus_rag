from fastapi import APIRouter
from app.api import auth, user, ingestion

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(user.router)
api_router.include_router(ingestion.router)