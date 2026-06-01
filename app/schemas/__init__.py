"""Pydantic schemas for the cycling training app."""

from datetime import datetime, date
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, EmailStr


# ── User ──

class UserCreate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    ftp: int = 200
    weight_kg: float = 75.0
    training_goal: str = "base"


class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    ftp: Optional[int] = None
    weight_kg: Optional[float] = None
    resting_hr: Optional[int] = None
    max_hr: Optional[int] = None
    training_goal: Optional[str] = None
    location_lat: Optional[float] = None
    location_lon: Optional[float] = None
    weather_preference: Optional[str] = None


class UserOut(BaseModel):
    id: int
    name: Optional[str] = None
    email: Optional[str] = None
    ftp: int
    weight_kg: float
    resting_hr: int
    max_hr: int
    training_goal: str
    strava_athlete_id: Optional[int] = None
    is_active: bool
    created_at: datetime
    location_lat: Optional[float] = None
    location_lon: Optional[float] = None
    weather_preference: str = "auto"

    class Config:
        from_attributes = True


# ── Strava Activity ──

class StravaActivityOut(BaseModel):
    id: int
    strava_id: int
    name: Optional[str] = None
    activity_type: str = "Ride"
    start_date: datetime
    elapsed_time: Optional[int] = None
    moving_time: Optional[int] = None
    distance: Optional[float] = None  # meters
    total_elevation_gain: Optional[float] = None
    average_watts: Optional[float] = None
    max_watts: Optional[float] = None
    weighted_average_watts: Optional[float] = None
    average_heartrate: Optional[float] = None
    max_heartrate: Optional[float] = None
    training_load: Optional[float] = None
    intensity_factor: Optional[float] = None
    training_stress_score: Optional[float] = None
    workout_type: Optional[str] = None

    class Config:
        from_attributes = True


# ── Workout ──

class WorkoutCreate(BaseModel):
    scheduled_date: date
    scheduled_time: Optional[str] = None
    workout_type: str = "endurance"
    title: str
    description: Optional[str] = None
    duration_minutes: Optional[int] = None
    target_power_zone: Optional[str] = None
    target_hr_zone: Optional[str] = None
    target_rpe: Optional[int] = None
    is_manual: bool = False


class WorkoutUpdate(BaseModel):
    status: Optional[str] = None
    scheduled_date: Optional[date] = None
    scheduled_time: Optional[str] = None
    actual_duration_minutes: Optional[int] = None
    actual_distance_km: Optional[float] = None
    actual_tss: Optional[float] = None
    actual_kj: Optional[float] = None
    notes: Optional[str] = None


class WorkoutOut(BaseModel):
    id: int
    user_id: int
    scheduled_date: date
    scheduled_time: Optional[str] = None
    workout_type: str
    title: str
    description: Optional[str] = None
    duration_minutes: Optional[int] = None
    target_power_zone: Optional[str] = None
    target_hr_zone: Optional[str] = None
    target_rpe: Optional[int] = None
    status: str
    is_manual: bool
    is_indoor: bool = False
    source: str
    actual_duration_minutes: Optional[int] = None
    actual_distance_km: Optional[float] = None
    actual_tss: Optional[float] = None
    actual_kj: Optional[float] = None
    notes: Optional[str] = None
    strava_activity_id: Optional[int] = None

    class Config:
        from_attributes = True


# ── Training Metrics ──

class TrainingMetricsOut(BaseModel):
    date: date
    ctl: float
    atl: float
    tsb: float
    total_tss: float
    total_duration_minutes: float
    total_distance_km: float
    ride_count: int

    class Config:
        from_attributes = True


# ── Dashboard ──

class DashboardResponse(BaseModel):
    current_ctl: float = 0
    current_atl: float = 0
    current_tsb: float = 0
    recent_activities_count: int = 0
    this_week_tss: float = 0
    suggested_workouts: List[WorkoutOut] = []
    training_goal: str = "base"
    ftp: int = 200
    strava_connected: bool = False


class PMCDataPoint(BaseModel):
    date: str
    ctl: float
    atl: float
    tsb: float


class PMCResponse(BaseModel):
    data: List[PMCDataPoint]


class WeeklyCalendarDay(BaseModel):
    date: str
    day_name: str
    workouts: List[WorkoutOut]
    total_tss: float = 0
    total_duration_minutes: float = 0
    weather: Optional[Dict] = None  # Weather info for this day


class WeeklyCalendarResponse(BaseModel):
    week_start: str
    week_end: str
    days: List[WeeklyCalendarDay]


# ── Strava Auth ──

class StravaAuthUrl(BaseModel):
    auth_url: str


class StravaTokenResponse(BaseModel):
    success: bool
    message: str


# ── Auth ──

class RegisterRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=6)
    name: Optional[str] = None
    ftp: Optional[int] = 200
    weight_kg: Optional[float] = 75.0


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: Optional[UserOut] = None


# ── Weather ──

class UserWeatherSettings(BaseModel):
    """User preferences for weather-aware training."""
    location_lat: Optional[float] = None
    location_lon: Optional[float] = None
    weather_preference: str = "auto"


class WeatherDayOut(BaseModel):
    """Weather forecast for a single day."""
    date: str
    symbol: str
    icon: str
    label: str
    temp_min: float
    temp_max: float
    precipitation_mm: float
    wind_speed_ms: float
    indoor_suitable: bool
    outdoor_suitable: bool
    suggestion: str
    suggestion_reason: str


class WeatherForecastResponse(BaseModel):
    """Weather forecast response."""
    location_lat: float
    location_lon: float
    days: List[WeatherDayOut]


# ── Recovery / Readiness ──

class RecoveryLogCreate(BaseModel):
    """Daily recovery check-in."""
    date: Optional[date] = None  # Defaults to today
    hrv_rmssd: Optional[float] = None
    sleep_hours: Optional[float] = None
    sleep_quality: Optional[int] = Field(None, ge=1, le=5)
    subjective_feeling: Optional[int] = Field(None, ge=1, le=10)
    soreness: Optional[int] = Field(None, ge=1, le=5)
    resting_hr: Optional[int] = None
    notes: Optional[str] = None


class RecoveryScoreOut(BaseModel):
    """Recovery score for a single day."""
    id: int
    date: date
    hrv_rmssd: Optional[float] = None
    sleep_hours: Optional[float] = None
    sleep_quality: Optional[int] = None
    subjective_feeling: Optional[int] = None
    soreness: Optional[int] = None
    readiness_score: float = 0
    readiness_zone: str = "yellow"
    resting_hr: Optional[int] = None
    notes: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ReadinessResponse(BaseModel):
    """Current readiness status with components."""
    readiness_score: float = 0
    readiness_zone: str = "yellow"
    components: Dict[str, float] = {}
    today_logged: bool = False
    streak_days: int = 0
    suggestion: str = ""
