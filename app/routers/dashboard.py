"""Dashboard router — aggregated views and training metrics."""

from datetime import date, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from app.database import get_db
from app.models import User, Workout, WorkoutStatus, StravaActivity, TrainingMetrics
from app.schemas import (
    DashboardResponse,
    PMCDataPoint,
    PMCResponse,
    TrainingMetricsOut,
    WeeklyCalendarDay,
    WeeklyCalendarResponse,
    WorkoutOut,
)
from app.auth import get_current_user, optional_current_user
from app.services.training_load import pmc_series

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/", response_model=DashboardResponse)
def get_dashboard(
    current_user: Optional[User] = Depends(optional_current_user),
    db: Session = Depends(get_db),
):
    """Get the main dashboard with current training metrics and suggested workouts."""
    if not current_user:
        return DashboardResponse()

    # Current training metrics
    latest_metrics = (
        db.query(TrainingMetrics)
        .filter(TrainingMetrics.user_id == current_user.id)
        .order_by(desc(TrainingMetrics.date))
        .first()
    )

    current_ctl = latest_metrics.ctl if latest_metrics else 0.0
    current_atl = latest_metrics.atl if latest_metrics else 0.0
    current_tsb = latest_metrics.tsb if latest_metrics else 0.0

    # Recent activities count (last 7 days)
    seven_days_ago = date.today() - timedelta(days=7)
    recent_count = (
        db.query(func.count(StravaActivity.id))
        .filter(
            StravaActivity.user_id == current_user.id,
            func.date(StravaActivity.start_date) >= seven_days_ago,
        )
        .scalar()
        or 0
    )

    # This week's TSS
    this_week_start = date.today() - timedelta(days=date.today().weekday())
    this_week_tss = (
        db.query(func.coalesce(func.sum(Workout.actual_tss), 0))
        .filter(
            Workout.user_id == current_user.id,
            Workout.scheduled_date >= this_week_start,
            Workout.status == WorkoutStatus.COMPLETED,
        )
        .scalar()
        or 0
    )

    # Suggested workouts (next 7 days)
    today = date.today()
    week_end = today + timedelta(days=7)
    suggested = (
        db.query(Workout)
        .filter(
            Workout.user_id == current_user.id,
            Workout.scheduled_date >= today,
            Workout.scheduled_date <= week_end,
            Workout.status.in_([WorkoutStatus.SUGGESTED, WorkoutStatus.ACCEPTED]),
        )
        .order_by(Workout.scheduled_date, Workout.scheduled_time)
        .all()
    )

    strava_connected = current_user.strava_access_token is not None

    return DashboardResponse(
        current_ctl=current_ctl,
        current_atl=current_atl,
        current_tsb=current_tsb,
        recent_activities_count=recent_count,
        this_week_tss=this_week_tss,
        suggested_workouts=suggested,
        training_goal=current_user.training_goal.value if hasattr(current_user.training_goal, 'value') else str(current_user.training_goal),
        ftp=current_user.ftp or 200,
        strava_connected=strava_connected,
    )


@router.get("/pmc", response_model=PMCResponse)
def get_pmc(
    days: int = Query(90, le=365, description="Number of days to include"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get Performance Management Chart data (CTL/ATL/TSB time series)."""
    thirty_days_ago = date.today() - timedelta(days=days)

    metrics_records = (
        db.query(TrainingMetrics)
        .filter(
            TrainingMetrics.user_id == current_user.id,
            TrainingMetrics.date >= thirty_days_ago,
        )
        .order_by(TrainingMetrics.date)
        .all()
    )

    if metrics_records:
        data = []
        for m in metrics_records:
            data.append(PMCDataPoint(
                date=m.date.isoformat(),
                ctl=m.ctl,
                atl=m.atl,
                tsb=m.tsb,
            ))
        return PMCResponse(data=data)

    # Fall back to computing from scratch
    daily_tss = {}
    workouts = (
        db.query(Workout)
        .filter(
            Workout.user_id == current_user.id,
            Workout.scheduled_date >= thirty_days_ago,
            Workout.status == WorkoutStatus.COMPLETED,
        )
        .all()
    )
    for w in workouts:
        tss = w.actual_tss or 0
        day = w.scheduled_date
        daily_tss[day] = daily_tss.get(day, 0) + tss

    activities = (
        db.query(StravaActivity)
        .filter(
            StravaActivity.user_id == current_user.id,
            func.date(StravaActivity.start_date) >= thirty_days_ago,
        )
        .all()
    )
    for a in activities:
        day = a.start_date.date()
        tss = a.training_stress_score or 0
        daily_tss[day] = max(daily_tss.get(day, 0), tss)

    series = pmc_series(daily_tss, days)
    return PMCResponse(data=[
        PMCDataPoint(date=d.isoformat(), ctl=ctl, atl=atl, tsb=tsb)
        for d, ctl, atl, tsb in series
    ])


@router.get("/calendar", response_model=WeeklyCalendarResponse)
def get_weekly_calendar(
    week_start: Optional[date] = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get the weekly calendar view with all workouts grouped by day."""
    if not week_start:
        today = date.today()
        week_start = today - timedelta(days=today.weekday())

    week_end = week_start + timedelta(days=6)

    workouts = (
        db.query(Workout)
        .filter(
            Workout.user_id == current_user.id,
            Workout.scheduled_date >= week_start,
            Workout.scheduled_date <= week_end,
        )
        .order_by(Workout.scheduled_date, Workout.scheduled_time)
        .all()
    )

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    days = []

    for i in range(7):
        day_date = week_start + timedelta(days=i)
        day_workouts = [w for w in workouts if w.scheduled_date == day_date]
        total_tss = sum(w.actual_tss or w.actual_tss or 0 for w in day_workouts)
        total_duration = sum(w.actual_duration_minutes or w.duration_minutes or 0 for w in day_workouts)

        days.append(WeeklyCalendarDay(
            date=day_date.isoformat(),
            day_name=day_names[i],
            workouts=day_workouts,
            total_tss=total_tss,
            total_duration_minutes=total_duration,
        ))

    return WeeklyCalendarResponse(
        week_start=week_start.isoformat(),
        week_end=week_end.isoformat(),
        days=days,
    )


@router.get("/metrics", response_model=List[TrainingMetricsOut])
def get_metrics(
    days: int = Query(30, le=365),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get historical training metrics."""
    cutoff = date.today() - timedelta(days=days)
    metrics = (
        db.query(TrainingMetrics)
        .filter(
            TrainingMetrics.user_id == current_user.id,
            TrainingMetrics.date >= cutoff,
        )
        .order_by(TrainingMetrics.date)
        .all()
    )
    return metrics
