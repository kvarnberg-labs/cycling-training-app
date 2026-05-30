"""User settings router."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.schemas import UserCreate, UserOut, UserUpdate

router = APIRouter(prefix="/user", tags=["user"])


def _get_first_user(db: Session) -> User:
    user = db.query(User).first()
    if not user:
        raise HTTPException(status_code=404, detail="No user found")
    return user


@router.get("/", response_model=UserOut)
def get_user(db: Session = Depends(get_db)):
    """Get current user profile."""
    return _get_first_user(db)


@router.post("/", response_model=UserOut, status_code=201)
def create_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Create a new user."""
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
def update_user(user_data: UserUpdate, db: Session = Depends(get_db)):
    """Update user settings (FTP, weight, goal, etc.)."""
    user = _get_first_user(db)
    update_dict = user_data.model_dump(exclude_unset=True)
    for key, value in update_dict.items():
        setattr(user, key, value)
    db.commit()
    db.refresh(user)
    return user
