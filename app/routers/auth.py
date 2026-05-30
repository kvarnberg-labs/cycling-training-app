"""Authentication router — register, login, token refresh."""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional

from app.database import get_db
from app.models import User
from app.auth import hash_password, verify_password, create_access_token, decode_token, get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])
templates = Jinja2Templates(directory="app/templates")


# ── Schemas ──

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    ftp: int = 200
    weight_kg: float = 75.0


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    name: str


class AuthStatus(BaseModel):
    authenticated: bool
    user_id: Optional[int] = None
    name: Optional[str] = None


# ── API Endpoints ──

@router.post("/register", response_model=TokenResponse)
def register(data: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user account."""
    # Check if email already exists
    existing = db.query(User).filter(User.email == data.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    user = User(
        name=data.name,
        email=data.email,
        password_hash=hash_password(data.password),
        ftp=data.ftp,
        weight_kg=data.weight_kg,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id, user.email)
    return TokenResponse(
        access_token=token,
        user_id=user.id,
        name=user.name or "",
    )


@router.post("/login", response_model=TokenResponse)
def login(data: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate and get an access token."""
    user = db.query(User).filter(User.email == data.email).first()
    if not user or not user.password_hash:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not verify_password(data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    token = create_access_token(user.id, user.email)
    return TokenResponse(
        access_token=token,
        user_id=user.id,
        name=user.name or "",
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh_token(
    current_user: User = Depends(get_current_user),
):
    """Refresh the access token for the current user."""
    token = create_access_token(current_user.id, current_user.email)
    return TokenResponse(
        access_token=token,
        user_id=current_user.id,
        name=current_user.name or "",
    )


@router.get("/status", response_model=AuthStatus)
def auth_status(
    current_user: Optional[User] = Depends(get_current_user),
):
    """Check authentication status."""
    if current_user is None:
        return AuthStatus(authenticated=False)
    return AuthStatus(
        authenticated=True,
        user_id=current_user.id,
        name=current_user.name,
    )


# ── Web UI Routes ──

@router.get("/login")
def login_page(request: Request):
    """Render the login/signup page."""
    return templates.TemplateResponse("login.html", {
        "request": request,
        "title": "Login — Cycling Training App",
    })


@router.get("/logout")
def logout():
    """Log out by redirecting to login (token-based, client-side removal)."""
    response = RedirectResponse(url="/auth/login", status_code=302)
    return response
