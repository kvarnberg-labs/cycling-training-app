"""Analytics router — workout history analytics and insights."""

from datetime import date, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, Workout, WorkoutStatus, TrainingMetrics
from app.auth import get_current_user

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/weekly")
def get_weekly_analytics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get weekly analytics for the last 12 weeks.

    Returns weekly TSS totals, workout type distribution, hours per week,
    and CTL/ATL change for each week.
    """
    today = date.today()
    current_week_start = today - timedelta(days=today.weekday())
    start_date = current_week_start - timedelta(weeks=12)

    # Query TrainingMetrics for the period
    metrics = (
        db.query(TrainingMetrics)
        .filter(
            TrainingMetrics.user_id == current_user.id,
            TrainingMetrics.date >= start_date,
        )
        .order_by(TrainingMetrics.date)
        .all()
    )
    metrics_by_date = {m.date: m for m in metrics}

    # Query completed workouts for type distribution
    workouts = (
        db.query(Workout)
        .filter(
            Workout.user_id == current_user.id,
            Workout.scheduled_date >= start_date,
            Workout.status == WorkoutStatus.COMPLETED,
        )
        .all()
    )

    weeks_data = []

    for i in range(12):
        week_start = current_week_start - timedelta(weeks=11 - i)
        week_end = week_start + timedelta(days=6)

        # Accumulate TSS and duration from TrainingMetrics
        week_tss = 0.0
        week_duration = 0.0
        for day_offset in range(7):
            d = week_start + timedelta(days=day_offset)
            if d in metrics_by_date:
                m = metrics_by_date[d]
                week_tss += m.total_tss or 0
                week_duration += m.total_duration_minutes or 0

        # Type distribution from completed workouts
        type_dist = {}
        for w in workouts:
            if week_start <= w.scheduled_date <= week_end:
                wt = (
                    w.workout_type.value
                    if hasattr(w.workout_type, "value")
                    else str(w.workout_type or "other")
                )
                type_dist[wt] = type_dist.get(wt, 0) + 1

        # CTL / ATL at start and end of week
        start_metric = metrics_by_date.get(week_start)
        end_metric = metrics_by_date.get(week_end)

        # Fallback: find closest metric before/at the date
        if not start_metric:
            for m in metrics:
                if m.date >= week_start:
                    start_metric = m
                    break

        if not end_metric:
            for m in reversed(metrics):
                if m.date <= week_end:
                    end_metric = m
                    break

        weeks_data.append({
            "week_start": week_start.isoformat(),
            "week_end": week_end.isoformat(),
            "total_tss": round(week_tss, 1),
            "total_hours": round(week_duration / 60, 1),
            "total_duration_minutes": int(week_duration),
            "ride_count": sum(
                1
                for w in workouts
                if week_start <= w.scheduled_date <= week_end
            ),
            "type_distribution": type_dist,
            "ctl_start": round(start_metric.ctl, 1) if start_metric else 0,
            "ctl_end": round(end_metric.ctl, 1) if end_metric else 0,
            "atl_start": round(start_metric.atl, 1) if start_metric else 0,
            "atl_end": round(end_metric.atl, 1) if end_metric else 0,
        })

    # Overall type distribution across all weeks
    overall_dist = {}
    for w in weeks_data:
        for wt, count in w["type_distribution"].items():
            overall_dist[wt] = overall_dist.get(wt, 0) + count

    # CTL / ATL trend for line chart (daily values)
    ctl_trend = [
        {
            "date": m.date.isoformat(),
            "ctl": round(m.ctl, 1),
            "atl": round(m.atl, 1),
        }
        for m in metrics
    ]

    return {
        "weeks": weeks_data,
        "overall_type_distribution": overall_dist,
        "ctl_trend": ctl_trend,
        "start_date": start_date.isoformat(),
        "end_date": current_week_start.isoformat(),
    }


@router.get("/power-curve")
def get_power_curve(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get the athlete's power-duration curve from Strava activities."""
    from app.services.power_curve import compute_power_curve, get_power_curve_trend
    ftp = current_user.ftp or 200
    curve = compute_power_curve(
        db=db,
        user_id=current_user.id,
        ftp=ftp,
        days_back=365,
    )
    trend = get_power_curve_trend(
        db=db,
        user_id=current_user.id,
        ftp=ftp,
    )
    return {**curve, "trend": trend}
