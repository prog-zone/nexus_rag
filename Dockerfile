FROM python:3.13-slim-bookworm

RUN groupadd -r appuser && useradd -r -g appuser appuser

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

COPY pyproject.toml .
RUN uv sync --no-install-project --no-dev

COPY . .

RUN uv sync --no-dev && \
    chown -R appuser:appuser /app

USER appuser

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--workers", "2", "--bind", "0.0.0.0:8000", "app.main:app"]