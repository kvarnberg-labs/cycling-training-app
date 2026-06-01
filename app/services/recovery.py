"""Recovery readiness service — computes readiness scores from HRV, sleep, and subjective data.

The readiness algorithm combines:
  - 40% TSB (Training Stress Balance) — how recovered is your training load
  - 30% Sleep quality — hours and self-reported quality
  - 20% Subjective feeling — how the user feels (1-10)
  - 10% HRV trend — heart rate variability trend (if available)

Score ranges:
  - 0-33  → Red (low readiness) → recovery day
  - 34-66 → Yellow (moderate readiness) → normal training
  - 67-100 → Green (high readiness) → can push hard
"""

import logging
from datetime import date, timedelta
from typing import Optional, Dict

from sqlalchemy.orm import Session

from app.models import User, RecoveryScore, TrainingMetrics
from app.schemas import RecoveryLogCreate

logger = logging.getLogger(__name__)

# ── Zone boundaries ──

READINESS_ZONES = [
    (0, 33, "red"),
    (34, 66, "yellow"),
    (67, 100, "green"),
]

ZONE_SUGGESTIONS = {
    "red": "Low readiness — take a recovery day or do a very light spin. Prioritise sleep and nutrition.",
    "yellow": "Moderate readiness — normal training is fine, but avoid overreaching. Listen to your body.",
    "green": "High readiness — you're well recovered! Good day for hard intervals, threshold work, or a long ride.",
}


def _score_tsb(tsb: float) -> float:
    """Score TSB component (max 40 points).

    TSB > +10 → well recovered
    TSB -10 to -30 → deep fatigue
    """
    if tsb >= 10:
        return 40.0
    elif tsb >= 5:
        return 35.0
    elif tsb >= 0:
        return 30.0
    elif tsb >= -5:
        return 20.0
    elif tsb >= -10:
        return 12.0
    elif tsb >= -20:
        return 6.0
    else:
        return 2.0


def _score_sleep(hours: float, quality: Optional[int]) -> float:
    """Score sleep component (max 30 points).

    15 points from hours + 15 points from quality.
    """
    # Hours: 15 points max
    if hours >= 8:
        hours_score = 15.0
    elif hours >= 7:
        hours_score = 12.0
    elif hours >= 6:
        hours_score = 8.0
    elif hours >= 5:
        hours_score = 4.0
    else:
        hours_score = 0.0

    # Quality: 15 points max (1-5 scale)
    if quality is None:
        quality_score = 7.5  # neutral if not reported
    elif quality >= 5:
        quality_score = 15.0
    elif quality >= 4:
        quality_score = 12.0
    elif quality >= 3:
        quality_score = 8.0
    elif quality >= 2:
        quality_score = 4.0
    else:
        quality_score = 0.0

    return hours_score + quality_score


def _score_subjective(feeling: int) -> float:
    """Score subjective feeling component (max 20 points).

    Maps 1-10 feeling scale to 0-20 points.
    """
    if feeling >= 10:
        return 20.0
    elif feeling >= 9:
        return 18.0
    elif feeling >= 8:
        return 16.0
    elif feeling >= 7:
        return 14.0
    elif feeling >= 6:
        return 11.0
    elif feeling >= 5:
        return 8.0
    elif feeling >= 4:
        return 5.0
    elif feeling >= 3:
        return 3.0
    elif feeling >= 2:
        return 1.0
    else:
        return 0.0


def _score_hrv(hrv_rmssd: Optional[float], recent_scores: list) -> float:
    """Score HRV component (max 10 points).

    Uses the latest HRV value and a 7-day trend if available.
    """
    # If we have no HRV data, give neutral score
    if hrv_rmssd is None:
        return 5.0

    # Basic HRV reading score (RMSSD)
    # Typical ranges: <30ms = low, 30-60 = normal, >60 = high
    if hrv_rmssd >= 60:
        hrv_score = 6.0
    elif hrv_rmssd >= 40:
        hrv_score = 5.0
    elif hrv_rmssd >= 25:
        hrv_score = 3.0
    else:
        hrv_score = 1.0

    # Trend bonus (if we have enough data)
    if len(recent_scores) >= 3:
        recent_hrv = [s.hrv_rmssd for s in recent_scores if s.hrv_rmssd is not None]
        if len(recent_hrv) >= 3:
            trend = recent_hrv[-1] - recent_hrv[0]
            if trend > 5:
                hrv_score += 4.0  # Improving HRV
            elif trend > 2:
                hrv_score += 2.0
            elif trend > -2:
                hrv_score += 2.0  # Stable
            # Declining gets no bonus

    return min(hrv_score, 10.0)


def _get_zone(score: float) -> str:
    """Get readiness zone from score."""
    for low, high, zone in READINESS_ZONES:
        if low <= score <= high:
            return zone
    return "yellow"


def compute_readiness(
    tsb: float,
    sleep_hours: Optional[float] = None,
    sleep_quality: Optional[int] = None,
    subjective_feeling: Optional[int] = None,
    hrv_rmssd: Optional[float] = None,
    recent_scores: Optional[list] = None,
) -> Dict:
    """Compute a composite readiness score from all available data.

    Args:
        tsb: Current Training Stress Balance from PMC
        sleep_hours: Hours of sleep last night
        sleep_quality: Self-reported sleep quality (1-5)
        subjective_feeling: Self-reported feeling (1-10)
        hrv_rmssd: Heart Rate Variability RMSSD in ms
        recent_scores: Recent RecoveryScore records for trend analysis

    Returns:
        Dict with readiness_score, readiness_zone, components
    """
    tsb_score = _score_tsb(tsb)

    sleep_score = 15.0  # neutral default
    if sleep_hours is not None:
        sleep_score = _score_sleep(sleep_hours, sleep_quality)

    subjective_score = 10.0  # neutral default
    if subjective_feeling is not None:
        subjective_score = _score_subjective(subjective_feeling)

    hrv_score = _score_hrv(hrv_rmssd, recent_scores or [])

    total = tsb_score + sleep_score + subjective_score + hrv_score
    total = max(0.0, min(100.0, total))

    zone = _get_zone(total)

    return {
        "readiness_score": round(total, 1),
        "readiness_zone": zone,
        "components": {
            "tsb": round(tsb_score, 1),
            "sleep": round(sleep_score, 1),
            "subjective": round(subjective_score, 1),
            "hrv": round(hrv_score, 1),
        },
        "suggestion": ZONE_SUGGESTIONS.get(zone, ""),
    }


def get_or_create_recovery_score(
    db: Session,
    user_id: int,
    target_date: date,
) -> RecoveryScore:
    """Get existing recovery score for a date, or create a new empty one."""
    existing = (
        db.query(RecoveryScore)
        .filter(
            RecoveryScore.user_id == user_id,
            RecoveryScore.date == target_date,
        )
        .first()
    )
    if existing:
        return existing

    score = RecoveryScore(user_id=user_id, date=target_date)
    db.add(score)
    db.flush()
    return score


def log_recovery(
    db: Session,
    user_id: int,
    data: RecoveryLogCreate,
) -> RecoveryScore:
    """Log a daily recovery check-in and compute readiness."""
    target_date = data.date or date.today()

    score = get_or_create_recovery_score(db, user_id, target_date)

    # Update fields
    if data.hrv_rmssd is not None:
        score.hrv_rmssd = data.hrv_rmssd
    if data.sleep_hours is not None:
        score.sleep_hours = data.sleep_hours
    if data.sleep_quality is not None:
        score.sleep_quality = data.sleep_quality
    if data.subjective_feeling is not None:
        score.subjective_feeling = data.subjective_feeling
    if data.soreness is not None:
        score.soreness = data.soreness
    if data.resting_hr is not None:
        score.resting_hr = data.resting_hr
    if data.notes is not None:
        score.notes = data.notes

    # Get TSB
    latest_metrics = (
        db.query(TrainingMetrics)
        .filter(TrainingMetrics.user_id == user_id)
        .order_by(TrainingMetrics.date.desc())
        .first()
    )
    tsb = latest_metrics.tsb if latest_metrics else 0.0

    # Get recent recovery scores for HRV trend
    recent = (
        db.query(RecoveryScore)
        .filter(
            RecoveryScore.user_id == user_id,
            RecoveryScore.date < target_date,
        )
        .order_by(RecoveryScore.date.desc())
        .limit(7)
        .all()
    )

    result = compute_readiness(
        tsb=tsb,
        sleep_hours=score.sleep_hours,
        sleep_quality=score.sleep_quality,
        subjective_feeling=score.subjective_feeling,
        hrv_rmssd=score.hrv_rmssd,
        recent_scores=recent,
    )

    score.readiness_score = result["readiness_score"]
    score.readiness_zone = result["readiness_zone"]
    db.commit()
    db.refresh(score)

    return score


def get_recovery_streak(db: Session, user_id: int) -> int:
    """Count consecutive days the user has logged recovery data."""
    today = date.today()
    streak = 0
    for i in range(365):  # Max 1 year streak
        check_date = today - timedelta(days=i)
        logged = (
            db.query(RecoveryScore)
            .filter(
                RecoveryScore.user_id == user_id,
                RecoveryScore.date == check_date,
                RecoveryScore.subjective_feeling.isnot(None),
            )
            .first()
        )
        if logged:
            streak += 1
        else:
            break
    return streak
