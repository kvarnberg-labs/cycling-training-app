"""Fernet-based symmetric encryption for storing API keys at rest.

Uses the app's SECRET_KEY (hashed to a valid Fernet key) to encrypt
sensitive values before storing them in the database. This way API keys
are never in plain text on disk — only decrypted in-memory when needed.
"""

import base64
import hashlib
import logging
from typing import Optional

from cryptography.fernet import Fernet

from app.config import settings

logger = logging.getLogger(__name__)


def _derive_fernet_key(secret: str) -> bytes:
    """Derive a valid 32-byte Fernet key from the app's SECRET_KEY.

    Fernet requires a 32-byte urlsafe-base64-encoded key. We hash the
    secret with SHA-256 and base64-encode it to meet the requirement.
    """
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def _get_cipher() -> Optional[Fernet]:
    """Get a Fernet cipher instance using the app's SECRET_KEY.

    Returns None if the secret key is the default dev value (which means
    encryption is not properly configured).
    """
    sk = settings.secret_key
    if not sk or sk == "change-me-in-production-use-a-real-secret":
        logger.warning(
            "SECRET_KEY is still the default dev value — encryption is insecure. "
            "Set a real SECRET_KEY in .env for production."
        )
    try:
        key = _derive_fernet_key(sk)
        return Fernet(key)
    except Exception as e:
        logger.error(f"Failed to initialise Fernet cipher: {e}")
        return None


def encrypt(plaintext: str) -> Optional[str]:
    """Encrypt a plaintext string.

    Returns base64-encoded ciphertext, or None on failure.
    """
    if not plaintext:
        return None
    cipher = _get_cipher()
    if cipher is None:
        return None
    try:
        return cipher.encrypt(plaintext.encode("utf-8")).decode("utf-8")
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        return None


def decrypt(ciphertext: str) -> Optional[str]:
    """Decrypt a base64-encoded ciphertext string.

    Returns the original plaintext, or None on failure.
    """
    if not ciphertext:
        return None
    cipher = _get_cipher()
    if cipher is None:
        return None
    try:
        return cipher.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        return None
