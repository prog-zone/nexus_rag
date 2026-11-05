"""Datetime helpers that tolerate both naive and aware values.

PostgreSQL returns timezone-aware datetimes for ``timestamptz`` columns, while
other backends (SQLite for tests, or a column created without a timezone) hand
back naive ones. Comparing the two raises ``TypeError: can't compare
offset-naive and offset-aware datetimes`` inside the request handler, which
surfaced as a 500 on the email-verification and password-reset paths.
"""

from datetime import datetime, timezone


def as_utc(value: datetime | None) -> datetime | None:
    """Return ``value`` as an aware UTC datetime, or None."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def is_expired(expires_at: datetime | None, now: datetime | None = None) -> bool:
    """True when ``expires_at`` is missing or in the past."""
    deadline = as_utc(expires_at)
    if deadline is None:
        return True
    return (as_utc(now) or datetime.now(timezone.utc)) > deadline
