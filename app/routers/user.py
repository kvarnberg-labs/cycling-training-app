"""User settings router."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.schemas import UserCreate, UserOut, UserUpdate
from app.auth import get_current_user

router = APIRouter(prefix="/user", tags=["user"])


@router.get("/", response_model=UserOut)
def get_user(current_user: User = Depends(get_current_user)):
    """Get current user profile."""
    return current_user


@router.post("/", response_model=UserOut, status_code=201)
def create_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Create a new user (legacy — prefer /auth/register)."""
    user = User(
        name=user_data.name,
        email=user_data.email,
        ftp=user_data.ftp,
        weight_kg=user_data.weight_kg,
        training_goal=user_data.training_goal,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.patch("/", response_model=UserOut)
def update_user(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update user settings (FTP, weight, goal, etc.)."""
    update_dict = user_data.model_dump(exclude_unset=True)
    for key, value in update_dict.items():
        setattr(current_user, key, value)
    db.commit()
    db.refresh(current_user)
    return current_user
