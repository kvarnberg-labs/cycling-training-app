"""Daily training metrics computation.

This runs as a background task to compute:
- Daily TSS totals from completed activities
- CTL/ATL/TSB for each day
- Updates the TrainingMetrics table
"""

from datetime import date, timedelta, datetime
import logging

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import User, StravaActivity, Workout, WorkoutStatus, TrainingMetrics
from app.services.training_load import exp_weighted_avg, TAU_CTL, TAU_ATL

logger = logging.getLogger(__name__)


def compute_daily_metrics_for_user(user: User, db: Session):
    """Compute training metrics for a user for the current day.

    Looks at completed workouts and synced Strava activities,
    calculates total TSS and duration, then updates CTL/ATL/TSB.
    """
    today = date.today()

    # Get total TSS from completed workouts today
    workout_tss = (
        db.query(func.coalesce(func.sum(Workout.actual_tss), 0))
        .filter(
            Workout.user_id == user.id,
            Workout.scheduled_date == today,
            Workout.status == WorkoutStatus.COMPLETED,
        )
        .scalar()
        or 0.0
    )

    # Get TSS from Strava activities today (not already linked to a completed workout)
    activity_tss = (
        db.query(func.coalesce(func.sum(StravaActivity.training_stress_score), 0))
        .filter(
            StravaActivity.user_id == user.id,
            func.date(StravaActivity.start_date) == today,
        )
        .scalar()
        or 0.0
    )

    total_tss = max(workout_tss, activity_tss)  # Don't double-count

    # Get total duration and distance
    total_duration = (
        db.query(func.coalesce(func.sum(Workout.actual_duration_minutes), 0))
        .filter(
            Workout.user_id == user.id,
            Workout.scheduled_date == today,
            Workout.status == WorkoutStatus.COMPLETED,
        )
        .scalar()
        or 0.0
    )

    total_distance = (
        db.query(func.coalesce(func.sum(StravaActivity.distance), 0))
        .filter(
            StravaActivity.user_id == user.id,
            func.date(StravaActivity.start_date) == today,
        )
        .scalar()
        or 0.0
    )
    total_distance_km = total_distance / 1000.0

    total_kj = (
        db.query(func.coalesce(func.sum(StravaActivity.kilojoules), 0))
        .filter(
            StravaActivity.user_id == user.id,
            func.date(StravaActivity.start_date) == today,
        )
        .scalar()
        or 0.0
    )

    ride_count = (
        db.query(func.count(StravaActivity.id))
        .filter(
            StravaActivity.user_id == user.id,
            StravaActivity.activity_type.in_(["Ride", "VirtualRide", "Zwift"]),
            func.date(StravaActivity.start_date) == today,
        )
        .scalar()
        or 0
    )

    # Get yesterday's metrics to compute exponential moving averages
    yesterday_metrics = (
        db.query(TrainingMetrics)
        .filter(
            TrainingMetrics.user_id == user.id,
            TrainingMetrics.date == today - timedelta(days=1),
        )
        .first()
    )

    prev_ctl = yesterday_metrics.ctl if yesterday_metrics else 0.0
    prev_atl = yesterday_metrics.atl if yesterday_metrics else 0.0

    # Calculate new CTL and ATL
    ctl = exp_weighted_avg(prev_ctl, total_tss, TAU_CTL)
    atl = exp_weighted_avg(prev_atl, total_tss, TAU_ATL)
    tsb = ctl - atl

    # Upsert today's metrics
    today_metrics = (
        db.query(TrainingMetrics)
        .filter(
            TrainingMetrics.user_id == user.id,
            TrainingMetrics.date == today,
        )
        .first()
    )

    if today_metrics:
        today_metrics.ctl = round(ctl, 1)
        today_metrics.atl = round(atl, 1)
        today_metrics.tsb = round(tsb, 1)
        today_metrics.total_tss = round(total_tss, 1)
        today_metrics.total_duration_minutes = total_duration
        today_metrics.total_distance_km = round(total_distance_km, 2)
        today_metrics.total_kj = round(total_kj, 1)
        today_metrics.ride_count = ride_count
    else:
        today_metrics = TrainingMetrics(
            user_id=user.id,
            date=today,
            ctl=round(ctl, 1),
            atl=round(atl, 1),
            tsb=round(tsb, 1),
            total_tss=round(total_tss, 1),
            total_duration_minutes=total_duration,
            total_distance_km=round(total_distance_km, 2),
            total_kj=round(total_kj, 1),
            ride_count=ride_count,
        )
        db.add(today_metrics)

    db.commit()
    logger.info(f"Metrics computed for user {user.id}: CTL={ctl:.1f}, ATL={atl:.1f}, TSB={tsb:.1f}")


def compute_all_users_metrics():
    """Compute training metrics for all active users."""
    db = SessionLocal()
    try:
        users = db.query(User).filter(User.is_active == True).all()
        for user in users:
            try:
                compute_daily_metrics_for_user(user, db)
            except Exception as e:
                logger.error(f"Failed to compute metrics for user {user.id}: {e}")
                db.rollback()
    finally:
        db.close()
