import asyncio
from sqlalchemy import delete
from datetime import datetime, timezone
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

from app.core.logger import log
from app.api.router import api_router
from app.core.config import settings
from app.core.database import engine
from app.core.limiter import limiter
from app.models.user import UserRefreshToken
from app.core.database import AsyncSessionLocal

async def cleanup_expired_tokens():
    """Background task to delete expired refresh tokens."""
    while True:
        try:
            # Sleep for 24 hours (86400 seconds)
            await asyncio.sleep(86400)
            
            async with AsyncSessionLocal() as session:
                query = delete(UserRefreshToken).where(
                    UserRefreshToken.expires_at < datetime.now(timezone.utc)
                )
                await session.execute(query)
                await session.commit()
            
            log.info("expired_tokens_cleaned_successfully")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error("token_cleanup_failed", error=str(e))


"""Manages app startup and shutdown events, including background tasks."""
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("app_starting", project=settings.PROJECT_NAME)
    cleanup_task = asyncio.create_task(cleanup_expired_tokens())
    yield
    cleanup_task.cancel()
    await engine.dispose()
    log.info("app_shutting_down")

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response


# Configure CORS for frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ARCHITECTURE NOTE: In a multi-worker setup (e.g., Gunicorn with 2+ workers), 
# all workers will trigger this background task simultaneously. This is intentional.
# PostgreSQL handles concurrent DELETE operations gracefully using row-level locks. 
# One worker will delete the rows, and the others will safely execute a 0-row delete.
# If scaling to massive traffic, consider migrating this to a dedicated cron job.

"""Formats rate limit errors into consistent JSON responses."""
def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    dummy_response = Response()
    dummy_response = request.app.state.limiter._inject_headers(
        dummy_response, request.state.view_rate_limit
    )
    
    retry_after = dummy_response.headers.get("retry-after", "60")
    client_ip = request.client.host if request.client else "unknown"
    
    log.warning("rate_limit_exceeded", path=request.url.path, ip=client_ip, wait_seconds=retry_after)

    return JSONResponse(
        status_code=429,
        content={
            "detail": "Too Many Requests",
            "message": f"Rate limit exceeded. Please try again in {retry_after} seconds.",
            "wait_seconds": int(retry_after)
        },
        headers={"Retry-After": retry_after}
    )

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, custom_rate_limit_exceeded_handler)


"""Catches duplicate database records or constraint violations gracefully."""
@app.exception_handler(IntegrityError)
async def sqlalchemy_integrity_error_handler(request: Request, exc: IntegrityError):
    log.warning("database_integrity_error", path=request.url.path, error=str(exc.__cause__))
    return JSONResponse(
        status_code=409,
        content={"detail": "Database conflict. This record might already exist or violates a constraint."}
    )


"""Safety net for unexpected crashes to prevent raw 500 HTML errors."""
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal server error occurred. Please try again later."}
    )


# Register all API endpoints under the /api/v1 prefix
app.include_router(api_router, prefix="/api/v1")

@app.get("/api/v1/health")
def health_check():
    """Simple health check endpoint to verify the API is running."""
    log.info("health_check_accessed")
    return {"message": f"{settings.PROJECT_NAME} API is running"}