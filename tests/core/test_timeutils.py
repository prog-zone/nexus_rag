"""Datetime helpers used by the OTP expiry checks."""

from datetime import datetime, timedelta, timezone

from app.core.timeutils import as_utc, is_expired


def test_aware_datetimes_are_compared_correctly():
    now = datetime.now(timezone.utc)

    assert is_expired(now - timedelta(minutes=1), now) is True
    assert is_expired(now + timedelta(minutes=1), now) is False


def test_naive_datetimes_are_treated_as_utc():
    """PostgreSQL returns aware values, SQLite returns naive ones."""
    now = datetime.now(timezone.utc)
    naive_past = (now - timedelta(minutes=5)).replace(tzinfo=None)
    naive_future = (now + timedelta(minutes=5)).replace(tzinfo=None)

    assert is_expired(naive_past, now) is True
    assert is_expired(naive_future, now) is False


def test_missing_expiry_counts_as_expired():
    assert is_expired(None) is True


def test_as_utc_normalises_both_shapes():
    naive = datetime(2026, 1, 1, 12, 0, 0)
    aware = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone(timedelta(hours=2)))

    assert as_utc(naive).tzinfo == timezone.utc
    assert as_utc(aware).hour == 10
    assert as_utc(None) is None


def test_expiry_uses_the_current_clock_by_default():
    assert is_expired(datetime.now(timezone.utc) - timedelta(seconds=1)) is True
    assert is_expired(datetime.now(timezone.utc) + timedelta(minutes=1)) is False
