from functools import lru_cache
from ipaddress import ip_address, ip_network

from fastapi import Request
from slowapi import Limiter

from app.core.config import settings


@lru_cache(maxsize=1)
def _trusted_networks():
    """Parse ``TRUSTED_PROXIES`` once. Invalid entries are ignored."""
    networks = []
    for entry in settings.TRUSTED_PROXIES.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            networks.append(ip_network(entry, strict=False))
        except ValueError:
            continue
    return tuple(networks)


def _is_trusted_proxy(host: str) -> bool:
    try:
        address = ip_address(host)
    except ValueError:
        return False
    return any(address in network for network in _trusted_networks())


def get_real_ip(request: Request) -> str:
    """Client address used as the rate-limit key.

    ``X-Forwarded-For`` is attacker-controlled: trusting it from anyone turned
    every per-IP limit (login, register, password reset) into a no-op, since a
    client could rotate the header per request. Only honour it when the request
    actually arrived from a configured reverse proxy.
    """
    peer = request.client.host if request.client else None

    if peer and _is_trusted_proxy(peer):
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client = forwarded.split(",")[0].strip()
            try:
                ip_address(client)
            except ValueError:
                return peer
            return client

    return peer or "unknown"


limiter = Limiter(key_func=get_real_ip)
