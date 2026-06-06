"""Intervals.icu router — connect, sync, and manage Intervals.icu data.

API keys are encrypted at rest in the database (per-user), never in plain
text files or config.
"""

import logging
from datetime import date, datetime, timedelta
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
from app.services.encryption import encrypt, decrypt

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/intervals", tags=["intervals"])


async def get_client_for_user(user: User) -> IntervalsClient:
    """Get an authenticated Intervals.icu client for a specific user.

    Decrypts the user's stored API key from the database. Falls back
    to the global ``settings.intervals_api_key`` for backwards compat
    (server-level config), but the per-user encrypted key takes priority.
    """
    api_key = None
    athlete_id = None

    # Per-user encrypted key (preferred — set via Settings UI)
    if user.intervals_api_key_encrypted:
        decrypted = decrypt(user.intervals_api_key_encrypted)
        if decrypted:
            api_key = decrypted
            athlete_id = user.intervals_athlete_id or settings.intervals_athlete_id
            logger.info(f"Using per-user encrypted API key for user {user.id}")

    # Fall back to server-level env config
    if not api_key:
        api_key = settings.intervals_api_key
        athlete_id = settings.intervals_athlete_id
        if api_key:
            logger.info("Using server-level INTERVALS_API_KEY from .env")

    if not api_key:
        raise HTTPException(
            status_code=400,
            detail=(
                "Intervals.icu not configured. "
                "Add your API key in Settings to store it encrypted in your account."
            ),
        )

    return IntervalsClient(api_key=api_key, athlete_id=athlete_id)


@router.get("/status")
async def get_status(
    current_user: User = Depends(get_current_user),
):
    """Check Intervals.icu connection status and athlete info."""
    # Quick check: do we have any key stored?
    has_key = bool(
        current_user.intervals_api_key_encrypted or settings.intervals_api_key
    )
    if not has_key:
        return {
            "connected": False,
            "message": "Not configured. Add your Intervals.icu API key in Settings.",
        }

    try:
        client = await get_client_for_user(current_user)
        athlete = await client.get_athlete()

        # Extract FTP from sportSettings (it's nested, not top-level)
        ftp = None
        sport_settings = athlete.get("sportSettings", [])
        for ss in sport_settings:
            if ss.get("types") and "Ride" in ss.get("types", []):
                ftp = ss.get("ftp")
                break
        if ftp is None and sport_settings:
            ftp = sport_settings[0].get("ftp")

        return {
            "connected": True,
            "athlete_name": athlete.get("name", "Unknown"),
            "athlete_id": athlete.get("id"),
            "ftp": ftp,
            "weight_kg": athlete.get("icu_weight"),
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
    client = await get_client_for_user(current_user)
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
    """Sync activities from Intervals.icu into our local database.
    Also auto-updates the user's FTP and weight from the athlete profile.
    """
    client = await get_client_for_user(current_user)

    try:
        activities = await client.get_activities(days_back=days_back, limit=200)
    except IntervalsError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # Auto-update user profile (FTP, weight) from Intervals.icu athlete data
    try:
        athlete = await client.get_athlete()
        updates = {}

        # Extract FTP from sportSettings (nested)
        sport_settings = athlete.get("sportSettings", [])
        for ss in sport_settings:
            if ss.get("types") and "Ride" in ss.get("types", []):
                ftp_from_intervals = ss.get("ftp")
                if ftp_from_intervals and ftp_from_intervals != current_user.ftp:
                    updates["ftp"] = ftp_from_intervals
                    logger.info(f"Updating FTP: {current_user.ftp} -> {ftp_from_intervals}")
                break

        # Extract weight
        weight = athlete.get("icu_weight") or athlete.get("weight")
        if weight and weight != current_user.weight_kg:
            updates["weight_kg"] = weight
            logger.info(f"Updating weight: {current_user.weight_kg} -> {weight}")

        # Extract resting HR
        resting_hr = athlete.get("icu_resting_hr")
        if resting_hr and resting_hr != current_user.resting_hr:
            updates["resting_hr"] = resting_hr
            logger.info(f"Updating resting HR: {current_user.resting_hr} -> {resting_hr}")

        # Extract max HR
        max_hr = athlete.get("athlete_max_hr")
        if max_hr and max_hr != current_user.max_hr:
            updates["max_hr"] = max_hr
            logger.info(f"Updating max HR: {current_user.max_hr} -> {max_hr}")

        if updates:
            for key, value in updates.items():
                setattr(current_user, key, value)
            db.commit()
            logger.info(f"Auto-updated user {current_user.id} profile from Intervals.icu: {updates}")

    except Exception as e:
        logger.warning(f"Could not auto-update profile from Intervals.icu: {e}")
        # Non-fatal — continue with sync

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
        # Use strava_id (integer) if available, otherwise hash the intervals string ID
        intervals_id_str = act.get("id", "")
        intervals_id = act.get("strava_id") or abs(hash(intervals_id_str))
        if not intervals_id or intervals_id in existing_ids:
            continue

        mapped = activity_to_dict(act)

        # Parse start_date from ISO string to datetime
        parsed_start = None
        raw_start = mapped.get("start_date")
        if raw_start:
            try:
                parsed_start = datetime.fromisoformat(raw_start.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        if not parsed_start:
            continue  # Skip activities without a valid start date

        # Create StravaActivity record (reusing the model for intervals data too)
        activity = StravaActivity(
            user_id=current_user.id,
            strava_id=intervals_id,  # Reuse strava_id for intervals.icu ID
            name=mapped.get("name", "Untitled"),
            activity_type=mapped.get("activity_type", "Ride"),
            start_date=parsed_start,
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
    client = await get_client_for_user(current_user)
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
    client = await get_client_for_user(current_user)
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


@router.post("/save-keys")
def save_intervals_keys(
    api_key: str = Query(..., description="Intervals.icu API key"),
    athlete_id: str = Query("", description="Intervals.icu athlete ID"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Save Intervals.icu credentials encrypted in the user's account.

    The API key is encrypted with Fernet (AES-128-CBC) using the app's
    SECRET_KEY before being stored. The athlete ID is stored in plain
    text (not sensitive — it's a public identifier).
    """
    encrypted = encrypt(api_key)
    if encrypted is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to encrypt API key. Check that SECRET_KEY is configured.",
        )

    current_user.intervals_api_key_encrypted = encrypted
    if athlete_id:
        current_user.intervals_athlete_id = athlete_id
    db.commit()

    logger.info(f"Saved encrypted Intervals.icu credentials for user {current_user.id}")
    return {
        "message": "Intervals.icu credentials saved securely (encrypted at rest).",
    }


@router.post("/clear-keys")
def clear_intervals_keys(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Remove stored Intervals.icu credentials from the user's account."""
    current_user.intervals_api_key_encrypted = None
    current_user.intervals_athlete_id = None
    db.commit()
    return {"message": "Intervals.icu credentials cleared."}


@router.get("/key-status")
def get_key_status(
    current_user: User = Depends(get_current_user),
):
    """Check whether the current user has Intervals.icu keys stored."""
    has_key = bool(current_user.intervals_api_key_encrypted)
    athlete_id = current_user.intervals_athlete_id or settings.intervals_athlete_id
    return {
        "has_key": has_key,
        "athlete_id": athlete_id or None,
        "source": "user" if has_key else ("server_config" if settings.intervals_api_key else None),
    }
