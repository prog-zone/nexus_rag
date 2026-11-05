import secrets
import string
from functools import lru_cache

from fastapi_mail import ConnectionConfig, FastMail, MessageSchema
from fastapi_mail.schemas import MessageType
from pydantic import EmailStr, SecretStr

from app.core.config import settings
from app.core.logger import log


@lru_cache(maxsize=1)
def get_mail_config() -> ConnectionConfig:
    """Build the SMTP config on first use.

    Building it at import time made ``import app.main`` fail on any checkout whose
    ``MAIL_FROM`` was not a valid address, so the API, the worker and the tests
    could not start without a fully configured mailbox.
    """
    return ConnectionConfig(
        MAIL_USERNAME=settings.MAIL_USERNAME,
        MAIL_PASSWORD=SecretStr(settings.MAIL_PASSWORD),
        MAIL_FROM=settings.MAIL_FROM,
        MAIL_PORT=settings.MAIL_PORT,
        MAIL_SERVER=settings.MAIL_HOST,
        MAIL_STARTTLS=True,
        MAIL_SSL_TLS=False,
        USE_CREDENTIALS=True,
    )


def is_mail_configured() -> bool:
    """True when enough mail settings exist to attempt a send."""
    return bool(settings.MAIL_FROM and settings.MAIL_HOST)


def generate_otp() -> str:
    return ''.join(secrets.choice(string.digits) for _ in range(6))

async def send_verification_email(email_to: EmailStr, otp: str):
    message = MessageSchema(
        subject="Verify your account",
        recipients=[email_to],  # type: ignore
        body=f"Your 6-digit verification code is: {otp}. It expires in 15 minutes.",
        subtype=MessageType.html
    )
    if not is_mail_configured():
        log.error("verification_email_skipped_no_mail_config", email=email_to)
        return
    fm = FastMail(get_mail_config())
    try:
        await fm.send_message(message)
        log.info("verification_email_sent", email=email_to)
    except Exception as e:
        log.error("verification_email_failed", email=email_to, error=str(e))


async def send_reset_password_email(email_to: EmailStr, otp: str):
    message = MessageSchema(
        subject="Password Reset Request",
        recipients=[email_to],  # type: ignore
        body=f"Your password reset code is: {otp}. It expires in 15 minutes. If you did not request this, please ignore this email.",
        subtype=MessageType.html
    )
    if not is_mail_configured():
        log.error("reset_password_email_skipped_no_mail_config", email=email_to)
        return
    fm = FastMail(get_mail_config())
    try:
        await fm.send_message(message)
        log.info("reset_password_email_sent", email=email_to)
    except Exception as e:
        log.error("reset_password_email_failed", email=email_to, error=str(e))
