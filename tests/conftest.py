"""Shared test fixtures and configuration."""

import os
import sys
import pytest
from datetime import date, datetime
from fastapi.testclient import TestClient

# Set test environment before importing app modules
os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["SECRET_KEY"] = "test-secret-key-for-testing"
os.environ["DEBUG"] = "false"

from app.database import Base, engine, SessionLocal, get_db
from app.main import app
from app.models import User, Workout, StravaActivity, TrainingMetrics, WorkoutType, TrainingGoal, WorkoutStatus
from app.auth import hash_password, create_access_token


@pytest.fixture(autouse=True)
def setup_database():
    """Create all tables before each test and drop them after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(setup_database):
    """Provide a clean database session for each test."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session):
    """FastAPI TestClient with overridden DB dependency."""

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def test_user(db_session):
    """Create a test user."""
    user = User(
        name="Test Rider",
        email="test@example.com",
        password_hash=hash_password("testpassword"),
        ftp=250,
        weight_kg=72.0,
        resting_hr=55,
        max_hr=190,
        training_goal=TrainingGoal.BASE,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers(test_user):
    """Generate authentication headers for the test user."""
    token = create_access_token(user_id=test_user.id, email=test_user.email)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_workout_data():
    """Sample workout creation data."""
    return {
        "scheduled_date": str(date.today() + __import__("datetime").timedelta(days=1)),
        "scheduled_time": "06:00",
        "workout_type": "endurance",
        "title": "Morning Endurance Ride",
        "description": "Steady Zone 2 effort",
        "duration_minutes": 90,
        "target_power_zone": "Zone 2 (56-75% FTP)",
        "target_rpe": 3,
    }
