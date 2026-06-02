"""Intervals.icu router — connect, sync, and manage Intervals.icu data."""

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import get_db
from app.models import User, StravaActivity, TrainingMetrics
from app.schemas import StravaActivityOut
from app.auth import get_current_user
from app.config import settings
from app.services.intervals_client import (
    IntervalsClient,
    IntervalsAuthError,
    IntervalsError,
    activity_to_dict,
    training_metrics_to_dict,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/intervals", tags=["intervals"])


async def get_client() -> IntervalsClient:
    """Get an authenticated Intervals.icu client or raise."""
    if not settings.intervals_api_key:
        raise HTTPException(
            status_code=400,
            detail="Intervals.icu not configured. Set INTERVALS_API_KEY in .env or in Settings.",
        )
    return IntervalsClient()


@router.get("/status")
async def get_status(
    current_user: User = Depends(get_current_user),
):
    """Check Intervals.icu connection status and athlete info."""
    if not settings.intervals_api_key:
        return {
            "connected": False,
            "message": "Not configured. Add your Intervals.icu API key in Settings.",
        }

    try:
        client = await get_client()
        athlete = await client.get_athlete()
        return {
            "connected": True,
            "athlete_name": athlete.get("name", "Unknown"),
            "athlete_id": athlete.get("id"),
            "ftp": athlete.get("ftp"),
            "weight_kg": athlete.get("weight"),
            "sports": athlete.get("sports", []),
            "message": "Connected to Intervals.icu",
        }
    except IntervalsAuthError as e:
        return {"connected": False, "message": str(e)}
    except Exception as e:
        logger.error(f"Intervals.icu status check failed: {e}")
        return {"connected": False, "message": f"Connection failed: {str(e)[:200]}"}


@router.get("/athlete")
async def get_athlete(
    current_user: User = Depends(get_current_user),
):
    """Get full athlete profile from Intervals.icu."""
    client = await get_client()
    try:
        athlete = await client.get_athlete()
        zones = await client.get_zones()
        return {
            "athlete": athlete,
            "zones": zones,
        }
    except IntervalsError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/sync")
async def sync_activities(
    days_back: int = Query(90, description="Number of days of activities to sync"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Sync activities from Intervals.icu into our local database."""
    client = await get_client()

    try:
        activities = await client.get_activities(days_back=days_back, limit=200)
    except IntervalsError as e:
        raise HTTPException(status_code=502, detail=str(e))

    if not activities:
        return {"synced": 0, "message": "No activities found in Intervals.icu"}

    # Get existing local IDs to deduplicate
    existing_ids = set(
        row[0]
        for row in db.query(StravaActivity.strava_id)
        .filter(StravaActivity.user_id == current_user.id)
        .all()
    )

    synced_count = 0
    ftp = current_user.ftp or 200

    for act in activities:
        intervals_id = act.get("id")
        if not intervals_id or intervals_id in existing_ids:
            continue

        mapped = activity_to_dict(act)

        # Create StravaActivity record (reusing the model for intervals data too)
        activity = StravaActivity(
            user_id=current_user.id,
            strava_id=intervals_id,  # Reuse strava_id for intervals.icu ID
            name=mapped.get("name", "Untitled"),
            activity_type=mapped.get("activity_type", "Ride"),
            start_date=mapped.get("start_date"),
            timezone=mapped.get("timezone"),
            elapsed_time=mapped.get("elapsed_time"),
            moving_time=mapped.get("moving_time"),
            distance=mapped.get("distance"),
            total_elevation_gain=mapped.get("total_elevation_gain"),
            average_watts=mapped.get("average_watts"),
            max_watts=mapped.get("max_watts"),
            weighted_average_watts=mapped.get("weighted_average_watts"),
            average_heartrate=mapped.get("average_heartrate"),
            max_heartrate=mapped.get("max_heartrate"),
            average_cadence=mapped.get("average_cadence"),
            kilojoules=mapped.get("kilojoules"),
            training_stress_score=mapped.get("training_stress_score"),
            intensity_factor=mapped.get("intensity_factor"),
            training_load=mapped.get("training_load"),
            workout_type=mapped.get("workout_type"),
        )
        db.add(activity)
        existing_ids.add(intervals_id)
        synced_count += 1

    db.commit()

    # Also pull training metrics if we have pre-computed data
    try:
        training_data = await client.get_training_metrics(days=days_back)
        for tm in training_data:
            mapped = training_metrics_to_dict(tm)
            metric_date_str = mapped.get("date")
            if not metric_date_str:
                continue
            try:
                metric_date = date.fromisoformat(
                    metric_date_str[:10] if isinstance(metric_date_str, str) else str(metric_date_str)[:10]
                )
            except (ValueError, TypeError):
                continue

            existing = (
                db.query(TrainingMetrics)
                .filter(
                    TrainingMetrics.user_id == current_user.id,
                    TrainingMetrics.date == metric_date,
                )
                .first()
            )
            if not existing:
                training_metric = TrainingMetrics(
                    user_id=current_user.id,
                    date=metric_date,
                    ctl=mapped.get("ctl", 0),
                    atl=mapped.get("atl", 0),
                    tsb=mapped.get("tsb", 0),
                    total_tss=mapped.get("total_tss", 0),
                    total_duration_minutes=mapped.get("total_duration_minutes", 0),
                    total_distance_km=mapped.get("total_distance_km", 0),
                )
                db.add(training_metric)

        db.commit()
    except Exception as e:
        logger.warning(f"Could not sync training metrics from Intervals.icu: {e}")

    return {
        "synced": synced_count,
        "total_intervals": len(activities),
        "message": f"Synced {synced_count} activities from Intervals.icu",
    }


@router.get("/activities", response_model=List[StravaActivityOut])
def list_synced_activities(
    limit: int = Query(50, le=200),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List Intervals.icu activities synced to our database."""
    activities = (
        db.query(StravaActivity)
        .filter(StravaActivity.user_id == current_user.id)
        .order_by(desc(StravaActivity.start_date))
        .limit(limit)
        .all()
    )
    return activities


@router.get("/power-curve")
async def get_power_curve(
    days: int = Query(365, le=730),
    current_user: User = Depends(get_current_user),
):
    """Get power-duration curve from Intervals.icu."""
    client = await get_client()
    try:
        curve = await client.get_power_curve(days=days)
        trend = await client.get_power_curve_trend()
        return {
            "curve": curve,
            "trend": trend,
            "source": "intervals.icu",
        }
    except IntervalsError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/training-metrics")
async def get_training_metrics(
    days: int = Query(90, le=365),
    current_user: User = Depends(get_current_user),
):
    """Get pre-computed training metrics (CTL/ATL/TSB) from Intervals.icu."""
    client = await get_client()
    try:
        metrics = await client.get_training_metrics(days=days)
        load = await client.get_training_load()
        return {
            "daily": metrics,
            "current": load,
            "source": "intervals.icu",
        }
    except IntervalsError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/update-settings")
def update_intervals_settings(
    api_key: str = Query(..., description="Intervals.icu API key"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Store Intervals.icu API key temporarily for this session.

    Note: For permanent storage, add INTERVALS_API_KEY to .env file.
    """
    settings.intervals_api_key = api_key
    return {"message": "Intervals.icu API key updated for this session"}
