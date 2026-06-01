"""Recovery readiness router — daily check-in and readiness score endpoints."""

from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, RecoveryScore
from app.schemas import RecoveryLogCreate, RecoveryScoreOut, ReadinessResponse
from app.auth import get_current_user
from app.services.recovery import (
    log_recovery,
    compute_readiness,
    get_recovery_streak,
)

router = APIRouter(prefix="/recovery", tags=["recovery"])


@router.post("/log", response_model=RecoveryScoreOut)
def log_recovery_checkin(
    data: RecoveryLogCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Log a daily recovery check-in (sleep, HRV, feeling, etc.).

    Computes and stores the readiness score automatically.
    """
    score = log_recovery(db, current_user.id, data)
    return score


@router.get("/readiness", response_model=ReadinessResponse)
def get_readiness(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get today's readiness score with component breakdown."""
    today = date.today()

    today_score = (
        db.query(RecoveryScore)
        .filter(
            RecoveryScore.user_id == current_user.id,
            RecoveryScore.date == today,
        )
        .first()
    )

    # If user has logged today, return that
    if today_score and today_score.readiness_score is not None:
        result = {
            "readiness_score": today_score.readiness_score,
            "readiness_zone": today_score.readiness_zone or "yellow",
            "components": {
                "tsb": 0,
                "sleep": 0,
                "subjective": 0,
                "hrv": 0,
            },
            "today_logged": True,
            "streak_days": get_recovery_streak(db, current_user.id),
            "suggestion": "",
        }
        return result

    # Otherwise compute from available data
    from app.models import TrainingMetrics
    latest_metrics = (
        db.query(TrainingMetrics)
        .filter(TrainingMetrics.user_id == current_user.id)
        .order_by(TrainingMetrics.date.desc())
        .first()
    )
    tsb = latest_metrics.tsb if latest_metrics else 0.0

    recent = (
        db.query(RecoveryScore)
        .filter(
            RecoveryScore.user_id == current_user.id,
            RecoveryScore.date < today,
        )
        .order_by(RecoveryScore.date.desc())
        .limit(7)
        .all()
    )

    readiness_kwargs = {
        "tsb": tsb,
        "sleep_hours": None,
        "sleep_quality": None,
        "subjective_feeling": None,
        "hrv_rmssd": None,
        "recent_scores": recent,
    }

    # If user has logged yesterday, use that sleep data
    if recent and recent[0].date == today - timedelta(days=1):
        yesterday = recent[0]
        readiness_kwargs["sleep_hours"] = yesterday.sleep_hours
        readiness_kwargs["sleep_quality"] = yesterday.sleep_quality

    result = compute_readiness(**readiness_kwargs)

    return ReadinessResponse(
        readiness_score=result["readiness_score"],
        readiness_zone=result["readiness_zone"],
        components=result["components"],
        today_logged=False,
        streak_days=get_recovery_streak(db, current_user.id),
        suggestion=result.get("suggestion", ""),
    )


@router.get("/history", response_model=list[RecoveryScoreOut])
def get_recovery_history(
    days: int = Query(30, le=90, description="Days of history"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get recovery score history."""
    cutoff = date.today() - timedelta(days=days)
    scores = (
        db.query(RecoveryScore)
        .filter(
            RecoveryScore.user_id == current_user.id,
            RecoveryScore.date >= cutoff,
        )
        .order_by(RecoveryScore.date.desc())
        .all()
    )
    return scores


@router.get("/today", response_model=Optional[RecoveryScoreOut])
def get_today_log(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get today's check-in if it exists."""
    today = date.today()
    score = (
        db.query(RecoveryScore)
        .filter(
            RecoveryScore.user_id == current_user.id,
            RecoveryScore.date == today,
        )
        .first()
    )
    return score
