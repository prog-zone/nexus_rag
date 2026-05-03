import taskiq_redis
from taskiq import TaskiqScheduler
from taskiq.schedule_sources import LabelScheduleSource
from taskiq_redis import RedisScheduleSource
from app.core.config import settings

# 1. Initialize Broker
broker = taskiq_redis.ListQueueBroker(settings.REDIS_URL)

# 2. Initialize Scheduler
scheduler = TaskiqScheduler(
    broker=broker,
    sources=[
        RedisScheduleSource(settings.REDIS_URL),
        LabelScheduleSource(broker),
        ],
)

# 3. Initialize FastAPI dependency injection
import taskiq_fastapi
taskiq_fastapi.init(broker, "app.main:app")

# 4. CRITICAL: Import tasks so they register with the broker
import app.tasks.ingestion
import app.tasks.maintenance