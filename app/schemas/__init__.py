"""Pydantic schemas for the cycling training app."""

from datetime import datetime, date
from typing import Optional, List
from pydantic import BaseModel, Field


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
