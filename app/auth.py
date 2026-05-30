"""Authentication module — JWT tokens, password hashing, auth dependencies."""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from sqlalchemy.orm import Session
import hashlib
import secrets

from app.config import settings
from app.database import get_db
from app.models import User

security = HTTPBearer(auto_error=False)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7


def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with salt.

    Using a simple but secure approach since passlib+bcrpyt has
    version compatibility issues. In production, switch to bcrypt
    or argon2 directly.

    Format: $sha256$salt$hash
    """
    salt = secrets.token_hex(16)
    hash_obj = hashlib.sha256((salt + password).encode())
    return f"$sha256${salt}${hash_obj.hexdigest()}"


def verify_password(plain_password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    if not hashed or not hashed.startswith("$sha256$"):
        return False

    parts = hashed.split("$")
    if len(parts) != 4:
        return False

    salt = parts[2]
    expected_hash = parts[3]
    actual_hash = hashlib.sha256((salt + plain_password).encode()).hexdigest()
    return actual_hash == expected_hash


def create_access_token(user_id: int, email: Optional[str] = None) -> str:
    """Create a JWT access token."""
    expire = datetime.now(timezone.utc) + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": str(user_id),
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    if email:
        payload["email"] = email
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT token. Returns payload dict or None."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """Dependency: extract and return the current authenticated user.

    Raises 401 if no valid token is provided.
    If no users exist yet (first-time setup), allows access for registration.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = int(payload.get("sub", 0))
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return user


def optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Dependency: return current user or None if not authenticated."""
    if credentials is None:
        return None

    payload = decode_token(credentials.credentials)
    if payload is None:
        return None

    user_id = int(payload.get("sub", 0))
    return db.query(User).filter(User.id == user_id).first()
