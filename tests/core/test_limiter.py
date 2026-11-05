"""Rate-limit key selection: X-Forwarded-For is only trusted from a proxy."""

from starlette.requests import Request

import pytest

from app.core import limiter as limiter_module
from app.core.limiter import get_real_ip


def _request(client_host: str | None, forwarded_for: str | None = None) -> Request:
    headers = []
    if forwarded_for is not None:
        headers.append((b"x-forwarded-for", forwarded_for.encode()))
    return Request({
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/login",
        "headers": headers,
        "client": (client_host, 12345) if client_host else None,
    })


@pytest.fixture(autouse=True)
def clear_trusted_cache():
    limiter_module._trusted_networks.cache_clear()
    yield
    limiter_module._trusted_networks.cache_clear()


def test_forwarded_for_is_ignored_from_an_untrusted_peer(monkeypatch):
    """A client rotating X-Forwarded-For must not get a fresh rate-limit bucket."""
    monkeypatch.setattr(limiter_module.settings, "TRUSTED_PROXIES", "")

    request = _request("203.0.113.9", forwarded_for="1.2.3.4")

    assert get_real_ip(request) == "203.0.113.9"


def test_forwarded_for_is_honoured_from_a_configured_proxy(monkeypatch):
    monkeypatch.setattr(limiter_module.settings, "TRUSTED_PROXIES", "10.0.0.1")
    limiter_module._trusted_networks.cache_clear()

    request = _request("10.0.0.1", forwarded_for="198.51.100.7, 10.0.0.1")

    assert get_real_ip(request) == "198.51.100.7"


def test_trusted_proxies_accept_cidr_blocks(monkeypatch):
    monkeypatch.setattr(limiter_module.settings, "TRUSTED_PROXIES", "10.0.0.0/8, 192.168.1.0/24")
    limiter_module._trusted_networks.cache_clear()

    assert get_real_ip(_request("10.4.5.6", "198.51.100.7")) == "198.51.100.7"
    assert get_real_ip(_request("192.168.1.10", "198.51.100.7")) == "198.51.100.7"
    assert get_real_ip(_request("172.16.0.1", "198.51.100.7")) == "172.16.0.1"


def test_a_malformed_forwarded_value_falls_back_to_the_peer(monkeypatch):
    monkeypatch.setattr(limiter_module.settings, "TRUSTED_PROXIES", "10.0.0.1")
    limiter_module._trusted_networks.cache_clear()

    assert get_real_ip(_request("10.0.0.1", "not-an-ip")) == "10.0.0.1"


def test_missing_client_falls_back_to_unknown(monkeypatch):
    monkeypatch.setattr(limiter_module.settings, "TRUSTED_PROXIES", "")

    assert get_real_ip(_request(None)) == "unknown"


def test_invalid_trusted_proxy_entries_are_ignored(monkeypatch):
    monkeypatch.setattr(limiter_module.settings, "TRUSTED_PROXIES", "nonsense, 10.0.0.1")
    limiter_module._trusted_networks.cache_clear()

    assert get_real_ip(_request("10.0.0.1", "198.51.100.7")) == "198.51.100.7"
    assert get_real_ip(_request("10.0.0.2", "198.51.100.7")) == "10.0.0.2"
