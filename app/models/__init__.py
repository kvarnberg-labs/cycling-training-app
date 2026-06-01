"""SQLAlchemy models for the cycling training app."""

from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Text, Boolean, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import relationship
import enum

from app.database import Base


class TrainingGoal(str, enum.Enum):
    BASE = "base"
    BUILD = "build"
    PEAK = "peak"
    RACE = "race"
    RECOVERY = "recovery"


class WorkoutType(str, enum.Enum):
    ENDURANCE = "endurance"
    TEMPO = "tempo"
    THRESHOLD = "threshold"
    VO2MAX = "vo2max"
    SPRINT = "sprint"
    RECOVERY = "recovery"
    INTERVAL = "interval"
    CUSTOM = "custom"


class WorkoutStatus(str, enum.Enum):
    SUGGESTED = "suggested"
    ACCEPTED = "accepted"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=True)
    email = Column(String(255), unique=True, nullable=True)
    password_hash = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Strava integration
    strava_athlete_id = Column(Integer, unique=True, nullable=True)
    strava_access_token = Column(String(255), nullable=True)
    strava_refresh_token = Column(String(255), nullable=True)
    strava_token_expires_at = Column(Integer, nullable=True)

    # User profile for training
    ftp = Column(Integer, default=200)  # Functional Threshold Power (watts)
    weight_kg = Column(Float, default=75.0)
    resting_hr = Column(Integer, default=60)
    max_hr = Column(Integer, default=185)
    training_goal = Column(SAEnum(TrainingGoal), default=TrainingGoal.BASE)
    is_active = Column(Boolean, default=True)

    # Weather-aware training settings
    location_lat = Column(Float, nullable=True)  # Latitude for weather forecast
    location_lon = Column(Float, nullable=True)  # Longitude for weather forecast
    weather_preference = Column(String(20), default="auto")  # "auto", "indoor", "outdoor"

    # Relationships
    strava_activities = relationship("StravaActivity", back_populates="user", cascade="all, delete-orphan")
    workouts = relationship("Workout", back_populates="user", cascade="all, delete-orphan")
    training_metrics = relationship("TrainingMetrics", back_populates="user", cascade="all, delete-orphan")


class StravaActivity(Base):
    __tablename__ = "strava_activities"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    strava_id = Column(Integer, unique=True, nullable=False)

    # Activity metadata
    name = Column(String(255), nullable=True)
    activity_type = Column(String(50), default="Ride")
    start_date = Column(DateTime, nullable=False)
    timezone = Column(String(50), nullable=True)
    elapsed_time = Column(Integer, nullable=True)  # seconds
    moving_time = Column(Integer, nullable=True)  # seconds
    distance = Column(Float, nullable=True)  # meters
    total_elevation_gain = Column(Float, nullable=True)  # meters

    # Power data
    average_watts = Column(Float, nullable=True)
    max_watts = Column(Float, nullable=True)
    weighted_average_watts = Column(Float, nullable=True)  # Normalized Power (NP)
    average_heartrate = Column(Float, nullable=True)
    max_heartrate = Column(Float, nullable=True)

    # Effort / load
    average_cadence = Column(Float, nullable=True)
    perceived_exertion = Column(Integer, nullable=True)  # 1-10 RPE
    kilojoules = Column(Float, nullable=True)
    suffer_score = Column(Integer, nullable=True)
    training_load = Column(Float, nullable=True)  # computed TSS or similar

    # Categorization
    workout_type = Column(SAEnum(WorkoutType), nullable=True)
    intensity_factor = Column(Float, nullable=True)  # IF = NP/FTP
    training_stress_score = Column(Float, nullable=True)  # TSS

    # Raw JSON for extensibility
    raw_data = Column(Text, nullable=True)  # JSON dump from Strava

    # Relationships
    user = relationship("User", back_populates="strava_activities")

    @property
    def hours(self) -> float:
        return (self.moving_time or 0) / 3600.0


class Workout(Base):
    """Suggested, planned, or manually logged workouts."""
    __tablename__ = "workouts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # When this workout should happen
    scheduled_date = Column(Date, nullable=False, index=True)
    scheduled_time = Column(String(10), nullable=True)  # e.g. "06:00"

    # What the workout is
    workout_type = Column(SAEnum(WorkoutType), nullable=False, default=WorkoutType.ENDURANCE)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    duration_minutes = Column(Integer, nullable=True)
    target_power_zone = Column(String(50), nullable=True)  # e.g. "Zone 2", "Sweet Spot"
    target_hr_zone = Column(String(50), nullable=True)
    target_rpe = Column(Integer, nullable=True)

    # Weather context
    is_indoor = Column(Boolean, default=False)  # True if weather forced indoor workout

    # Status tracking
    status = Column(SAEnum(WorkoutStatus), default=WorkoutStatus.SUGGESTED)
    is_manual = Column(Boolean, default=False)  # True if user logged it manually
    source = Column(String(50), default="recommendation")  # "recommendation", "manual", "strava_import"

    # If completed, actual metrics
    actual_duration_minutes = Column(Integer, nullable=True)
    actual_distance_km = Column(Float, nullable=True)
    actual_tss = Column(Float, nullable=True)
    actual_kj = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)

    # Link to Strava activity if imported
    strava_activity_id = Column(Integer, ForeignKey("strava_activities.id"), nullable=True)

    # Relationships
    user = relationship("User", back_populates="workouts")
    strava_activity = relationship("StravaActivity")


class TrainingMetrics(Base):
    """Daily training metrics — CTL, ATL, TSB."""
    __tablename__ = "training_metrics"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)

    # Performance Management Chart metrics (Banister model)
    ctl = Column(Float, default=0.0)  # Chronic Training Load (fitness) — 42-day weighted avg
    atl = Column(Float, default=0.0)  # Acute Training Load (fatigue) — 7-day weighted avg
    tsb = Column(Float, default=0.0)  # Training Stress Balance (form) — CTL - ATL

    # Additional daily stats
    total_tss = Column(Float, default=0.0)  # Total TSS for this day
    total_duration_minutes = Column(Float, default=0.0)
    total_distance_km = Column(Float, default=0.0)
    total_kj = Column(Float, default=0.0)
    ride_count = Column(Integer, default=0)

    # Relationships
    user = relationship("User", back_populates="training_metrics")
