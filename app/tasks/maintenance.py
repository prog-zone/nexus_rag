from datetime import datetime, timezone
from sqlalchemy import delete
from app.tkq import broker
from app.core.database import AsyncSessionLocal
from app.models.user import UserRefreshToken
from app.core.logger import log


@broker.task(schedule=[{"cron": "0 3 * * *"}])
async def cleanup_expired_tokens():
    async with AsyncSessionLocal() as session:
        try:
            query = delete(UserRefreshToken).where(
                UserRefreshToken.expires_at < datetime.now(timezone.utc)
            )
            result = await session.execute(query)
            await session.commit()
            log.info("expired_tokens_cleaned", deleted_count=result.rowcount)   # type: ignore
        except Exception as e:
            await session.rollback()
            log.error("token_cleanup_failed", error=str(e))
            