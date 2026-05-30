"""Workouts router — CRUD for planned and logged workouts."""

from datetime import date, timedelta, datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import get_db
from app.models import User, Workout, WorkoutStatus, StravaActivity
from app.schemas import WorkoutCreate, WorkoutOut, WorkoutUpdate
from app.services.recommendation_engine import generate_weekly_plan
from app.services.training_load import (
    pmc_series,
    calculate_tss,
    exp_weighted_avg,
)

router = APIRouter(prefix="/workouts", tags=["workouts"])


def _get_first_user(db: Session) -> Optional[User]:
    return db.query(User).first()


@router.get("/", response_model=List[WorkoutOut])
def list_workouts(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List workouts with optional date range and status filters."""
    user = _get_first_user(db)
    if not user:
        return []

    query = db.query(Workout).filter(Workout.user_id == user.id)

    if start_date:
        query = query.filter(Workout.scheduled_date >= start_date)
    if end_date:
        query = query.filter(Workout.scheduled_date <= end_date)
    if status:
        query = query.filter(Workout.status == status)

    return query.order_by(Workout.scheduled_date, Workout.scheduled_time).all()


@router.post("/", response_model=WorkoutOut, status_code=201)
def create_workout(
    workout_data: WorkoutCreate,
    db: Session = Depends(get_db),
):
    """Create a new workout (manual log or scheduled workout)."""
    user = _get_first_user(db)
    if not user:
        raise HTTPException(status_code=400, detail="No user found")

    workout = Workout(
        user_id=user.id,
        scheduled_date=workout_data.scheduled_date,
        scheduled_time=workout_data.scheduled_time,
        workout_type=workout_data.workout_type,
        title=workout_data.title,
        description=workout_data.description,
        duration_minutes=workout_data.duration_minutes,
        target_power_zone=workout_data.target_power_zone,
        target_hr_zone=workout_data.target_hr_zone,
        target_rpe=workout_data.target_rpe,
        is_manual=workout_data.is_manual,
        status=WorkoutStatus.SUGGESTED if not workout_data.is_manual else WorkoutStatus.COMPLETED,
        source="manual" if workout_data.is_manual else "recommendation",
    )

    db.add(workout)
    db.commit()
    db.refresh(workout)
    return workout


@router.get("/{workout_id}", response_model=WorkoutOut)
def get_workout(workout_id: int, db: Session = Depends(get_db)):
    """Get a single workout by ID."""
    workout = db.query(Workout).filter(Workout.id == workout_id).first()
    if not workout:
        raise HTTPException(status_code=404, detail="Workout not found")
    return workout


@router.patch("/{workout_id}", response_model=WorkoutOut)
def update_workout(
    workout_id: int,
    update_data: WorkoutUpdate,
    db: Session = Depends(get_db),
):
    """Update a workout (change status, log results, etc.)."""
    workout = db.query(Workout).filter(Workout.id == workout_id).first()
    if not workout:
        raise HTTPException(status_code=404, detail="Workout not found")

    update_dict = update_data.model_dump(exclude_unset=True)
    for key, value in update_dict.items():
        setattr(workout, key, value)

    db.commit()
    db.refresh(workout)
    return workout


@router.delete("/{workout_id}", status_code=204)
def delete_workout(workout_id: int, db: Session = Depends(get_db)):
    """Delete a workout."""
    workout = db.query(Workout).filter(Workout.id == workout_id).first()
    if not workout:
        raise HTTPException(status_code=404, detail="Workout not found")
    db.delete(workout)
    db.commit()


@router.post("/generate-week", response_model=List[WorkoutOut])
def generate_week(
    week_start: date = Query(default=None, description="Monday of the target week"),
    db: Session = Depends(get_db),
):
    """Generate workout recommendations for a week.

    Analyzes training load, recent workouts, and the user's goal
    to recommend a week of structured workouts.
    """
    user = _get_first_user(db)
    if not user:
        raise HTTPException(status_code=400, detail="No user found")

    if not week_start:
        today = date.today()
        week_start = today - timedelta(days=today.weekday())

    week_end = week_start + timedelta(days=6)

    # Get current training metrics
    from app.models import TrainingMetrics

    latest_metrics = (
        db.query(TrainingMetrics)
        .filter(TrainingMetrics.user_id == user.id)
        .order_by(desc(TrainingMetrics.date))
        .first()
    )

    ctl = latest_metrics.ctl if latest_metrics else 0.0
    atl = latest_metrics.atl if latest_metrics else 0.0
    tsb = latest_metrics.tsb if latest_metrics else 0.0

    # Get recent workouts (last 30 days for workout type analysis)
    thirty_days_ago = date.today() - timedelta(days=30)
    recent_workouts = (
        db.query(Workout)
        .filter(
            Workout.user_id == user.id,
            Workout.scheduled_date >= thirty_days_ago,
            Workout.status == WorkoutStatus.COMPLETED,
        )
        .all()
    )

    # Get existing scheduled workouts for this week
    existing = (
        db.query(Workout)
        .filter(
            Workout.user_id == user.id,
            Workout.scheduled_date >= week_start,
            Workout.scheduled_date <= week_end,
        )
        .all()
    )

    # Generate recommendations
    recommendations = generate_weekly_plan(
        user_id=user.id,
        ctl=ctl,
        atl=atl,
        tsb=tsb,
        goal=user.training_goal,
        ftp=user.ftp or 200,
        recent_workouts=recent_workouts,
        existing_scheduled=existing,
        week_start=week_start,
    )

    # Save to DB
    created_workouts = []
    for rec in recommendations:
        workout = Workout(**rec)
        db.add(workout)
        db.flush()
        db.refresh(workout)
        created_workouts.append(workout)

    db.commit()
    return created_workouts


@router.put("/{workout_id}/accept", response_model=WorkoutOut)
def accept_workout(workout_id: int, db: Session = Depends(get_db)):
    """Accept a suggested workout (moves it from suggested to accepted)."""
    workout = db.query(Workout).filter(Workout.id == workout_id).first()
    if not workout:
        raise HTTPException(status_code=404, detail="Workout not found")
    workout.status = WorkoutStatus.ACCEPTED
    db.commit()
    db.refresh(workout)
    return workout


@router.put("/{workout_id}/complete", response_model=WorkoutOut)
def complete_workout(workout_id: int, db: Session = Depends(get_db)):
    """Mark a workout as completed."""
    workout = db.query(Workout).filter(Workout.id == workout_id).first()
    if not workout:
        raise HTTPException(status_code=404, detail="Workout not found")
    workout.status = WorkoutStatus.COMPLETED
    db.commit()
    db.refresh(workout)
    return workout


@router.put("/{workout_id}/skip", response_model=WorkoutOut)
def skip_workout(workout_id: int, db: Session = Depends(get_db)):
    """Skip a workout."""
    workout = db.query(Workout).filter(Workout.id == workout_id).first()
    if not workout:
        raise HTTPException(status_code=404, detail="Workout not found")
    workout.status = WorkoutStatus.SKIPPED
    db.commit()
    db.refresh(workout)
    return workout
