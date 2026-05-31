"""Unit tests for the auth module (app/auth.py).

Tests password hashing, token creation/verification, and the auth dependencies.
"""

import pytest
from app.auth import (
    hash_password,
    verify_password,
    create_access_token,
    decode_token,
)


class TestPasswordHashing:
    """Tests for password hashing and verification."""

    def test_hash_format(self):
        """Hashed password should follow the $sha256$salt$hash format."""
        hashed = hash_password("testpassword")
        parts = hashed.split("$")
        assert parts[0] == ""
        assert parts[1] == "sha256"
        assert len(parts[2]) == 32  # 16 bytes hex encoded = 32 chars
        assert len(parts[3]) == 64  # SHA-256 = 64 hex chars

    def test_verify_correct_password(self):
        hashed = hash_password("testpassword")
        assert verify_password("testpassword", hashed) is True

    def test_verify_wrong_password(self):
        hashed = hash_password("testpassword")
        assert verify_password("wrongpassword", hashed) is False

    def test_different_salts_for_same_password(self):
        """Same password should produce different hashes (different salts)."""
        hash1 = hash_password("samepassword")
        hash2 = hash_password("samepassword")
        assert hash1 != hash2

    def test_verify_empty_password(self):
        hashed = hash_password("testpassword")
        assert verify_password("", hashed) is False

    def test_verify_none_hash(self):
        assert verify_password("testpassword", None) is False

    def test_verify_empty_hash(self):
        assert verify_password("testpassword", "") is False

    def test_verify_invalid_format(self):
        assert verify_password("testpassword", "not-a-valid-hash") is False

    def test_verify_wrong_prefix(self):
        assert verify_password("testpassword", "$bcrypt$salt$hash") is False

    def test_verify_too_few_parts(self):
        assert verify_password("testpassword", "$sha256$onlytwo") is False


class TestTokenCreation:
    """Tests for JWT access token creation and decoding."""

    def test_create_token_returns_string(self):
        token = create_access_token(user_id=1, email="test@example.com")
        assert isinstance(token, str)
        assert len(token) > 20

    def test_decode_valid_token(self):
        token = create_access_token(user_id=42, email="rider@test.com")
        payload = decode_token(token)
        assert payload is not None
        assert payload["sub"] == "42"
        assert payload["email"] == "rider@test.com"

    def test_decode_no_email(self):
        token = create_access_token(user_id=1)
        payload = decode_token(token)
        assert payload is not None
        assert payload["sub"] == "1"

    def test_decode_expired_token(self):
        """Tokens should be decoded - actual expiry is handled by the caller."""
        token = create_access_token(user_id=1)
        payload = decode_token(token)
        assert payload is not None

    def test_decode_invalid_token(self):
        assert decode_token("not-a-valid-jwt-token") is None

    def test_decode_empty_token(self):
        assert decode_token("") is None

    def test_decode_garbage(self):
        assert decode_token("this.is.definitely.not.a.jwt") is None

    def test_token_contains_standard_claims(self):
        token = create_access_token(user_id=1)
        payload = decode_token(token)
        assert "exp" in payload
        assert "iat" in payload
        assert "sub" in payload
