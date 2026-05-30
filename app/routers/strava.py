"""Strava OAuth and activity sync router."""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import get_db
from app.models import User, StravaActivity, Workout, WorkoutStatus
from app.schemas import StravaAuthUrl, StravaTokenResponse, StravaActivityOut
from app.services.strava_client import (
    StravaClient,
    exchange_authorization_code,
    get_strava_oauth_url,
    strava_activity_to_model,
)
from app.services.training_load import calculate_tss, classify_workout_type

router = APIRouter(prefix="/strava", tags=["strava"])


@router.get("/auth-url", response_model=StravaAuthUrl)
def get_auth_url():
    """Get the Strava OAuth authorization URL."""
    return StravaAuthUrl(auth_url=get_strava_oauth_url())


@router.get("/callback", response_model=StravaTokenResponse)
def strava_callback(
    code: str = Query(...),
    db: Session = Depends(get_db),
):
    """Handle the Strava OAuth callback.

    Exchange the authorization code for tokens and store them.
    For simplicity in this single-user app, updates the first active user.
    """
    token_data = exchange_authorization_code(code)
    if not token_data:
        raise HTTPException(status_code=400, detail="Failed to exchange authorization code")

    athlete = token_data.get("athlete", {})
    strava_athlete_id = athlete.get("id")
    athlete_name = f"{athlete.get('firstname', '')} {athlete.get('lastname', '')}".strip()

    # Find or create user
    user = db.query(User).filter(User.strava_athlete_id == strava_athlete_id).first()
    if not user:
        user = db.query(User).first()  # Single-user mode: use first user

    if not user:
        # Create a new user
        user = User(
            name=athlete_name or "Cyclist",
            strava_athlete_id=strava_athlete_id,
        )
        db.add(user)
        db.flush()

    # Update Strava tokens
    user.strava_athlete_id = strava_athlete_id
    user.strava_access_token = token_data["access_token"]
    user.strava_refresh_token = token_data["refresh_token"]
    user.strava_token_expires_at = token_data.get("expires_at", 0)
    if athlete_name:
        user.name = athlete_name

    db.commit()

    return StravaTokenResponse(
        success=True,
        message=f"Strava account connected! Athlete: {athlete_name}",
    )


@router.post("/sync", response_model=dict)
def sync_strava_activities(
    days_back: int = Query(90, description="Number of days of activities to sync"),
    db: Session = Depends(get_db),
):
    """Sync recent Strava activities for the connected user."""
    user = db.query(User).filter(User.strava_access_token.isnot(None)).first()
    if not user:
        raise HTTPException(status_code=400, detail="No Strava-connected user found")

    client = StravaClient(user.strava_access_token)

    # Check token expiry and refresh if needed
    if user.strava_token_expires_at and user.strava_token_expires_at < datetime.utcnow().timestamp():
        token_data = client.refresh_token(user.strava_refresh_token)
        if token_data:
            user.strava_access_token = token_data["access_token"]
            user.strava_refresh_token = token_data["refresh_token"]
            user.strava_token_expires_at = token_data.get("expires_at", 0)
            db.commit()
            # Recreate client with new token
            client = StravaClient(user.strava_access_token)
        else:
            raise HTTPException(status_code=401, detail="Failed to refresh Strava token")

    # Fetch activities
    after_date = datetime.utcnow() - timedelta(days=days_back)
    all_activities = []
    page = 1

    while True:
        activities = client.get_activities(page=page, after=after_date)
        if not activities:
            break
        all_activities.extend(activities)
        page += 1
        if len(activities) < 30:
            break

    if not all_activities:
        return {"synced": 0, "message": "No new activities found"}

    # Get existing Strava IDs to avoid duplicates
    existing_ids = set(
        row[0] for row in db.query(StravaActivity.strava_id).filter(
            StravaActivity.user_id == user.id
        ).all()
    )

    synced_count = 0
    ftp = user.ftp or 200

    for activity_data in all_activities:
        strava_id = activity_data["id"]
        if strava_id in existing_ids:
            continue

        # Only process Ride and VirtualRide activities
        activity_type = activity_data.get("type", "")
        if activity_type not in ("Ride", "VirtualRide", "Zwift"):
            continue

        activity = strava_activity_to_model(user.id, activity_data, ftp)
        db.add(activity)
        existing_ids.add(strava_id)
        synced_count += 1

    db.commit()

    return {
        "synced": synced_count,
        "total_strava": len(all_activities),
        "message": f"Synced {synced_count} new activities",
    }


@router.get("/activities", response_model=list[StravaActivityOut])
def list_strava_activities(
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db),
):
    """List synced Strava activities."""
    user = db.query(User).filter(User.strava_access_token.isnot(None)).first()
    if not user:
        return []

    activities = (
        db.query(StravaActivity)
        .filter(StravaActivity.user_id == user.id)
        .order_by(desc(StravaActivity.start_date))
        .limit(limit)
        .all()
    )
    return activities
