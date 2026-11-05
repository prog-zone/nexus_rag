"""Per-IP rate limits must not be bypassable by a spoofed X-Forwarded-For."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.core.limiter import limiter


@pytest.mark.asyncio
async def test_login_limit_cannot_be_dodged_by_rotating_forwarded_for():
    """Six attempts from one address, each claiming a different client IP."""
    limiter.reset()
    transport = ASGITransport(app=app, client=("203.0.113.50", 1234))

    async with AsyncClient(transport=transport, base_url="https://testserver") as client:
        statuses = []
        for attempt in range(6):
            response = await client.post(
                "/api/v1/auth/login",
                data={"username": "nobody@example.com", "password": "wrong-password"},
                headers={"X-Forwarded-For": f"198.51.100.{attempt}"},
            )
            statuses.append(response.status_code)

    limiter.reset()

    assert statuses[:5] == [400] * 5
    assert statuses[5] == 429


@pytest.mark.asyncio
async def test_same_forwarded_ip_still_shares_one_bucket(monkeypatch):
    """A proxy that is trusted forwards the real client, which keeps the limit."""
    from app.core import limiter as limiter_module

    monkeypatch.setattr(limiter_module.settings, "TRUSTED_PROXIES", "10.0.0.1")
    limiter_module._trusted_networks.cache_clear()
    limiter.reset()

    transport = ASGITransport(app=app, client=("10.0.0.1", 1234))
    try:
        async with AsyncClient(transport=transport, base_url="https://testserver") as client:
            statuses = []
            for _ in range(6):
                response = await client.post(
                    "/api/v1/auth/login",
                    data={"username": "nobody@example.com", "password": "wrong-password"},
                    headers={"X-Forwarded-For": "198.51.100.77"},
                )
                statuses.append(response.status_code)
    finally:
        limiter.reset()
        limiter_module._trusted_networks.cache_clear()

    assert statuses[:5] == [400] * 5
    assert statuses[5] == 429
