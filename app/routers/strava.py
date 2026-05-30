"""Strava OAuth and activity sync router."""

from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import get_db
from app.models import User, StravaActivity, Workout, WorkoutStatus
from app.schemas import StravaAuthUrl, StravaTokenResponse, StravaActivityOut
from app.auth import get_current_user
from app.services.strava_client import (
    StravaClient,
    exchange_authorization_code,
    get_strava_oauth_url,
    strava_activity_to_model,
)

router = APIRouter(prefix="/strava", tags=["strava"])


@router.get("/auth-url", response_model=StravaAuthUrl)
def get_auth_url():
    """Get the Strava OAuth authorization URL."""
    return StravaAuthUrl(auth_url=get_strava_oauth_url())


@router.get("/callback", response_model=StravaTokenResponse)
def strava_callback(
    code: str = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Handle the Strava OAuth callback."""
    token_data = exchange_authorization_code(code)
    if not token_data:
        raise HTTPException(status_code=400, detail="Failed to exchange authorization code")

    athlete = token_data.get("athlete", {})
    strava_athlete_id = athlete.get("id")

    # Update current user's Strava tokens
    current_user.strava_athlete_id = strava_athlete_id
    current_user.strava_access_token = token_data["access_token"]
    current_user.strava_refresh_token = token_data["refresh_token"]
    current_user.strava_token_expires_at = token_data.get("expires_at", 0)
    athlete_name = f"{athlete.get('firstname', '')} {athlete.get('lastname', '')}".strip()
    if athlete_name:
        current_user.name = athlete_name

    db.commit()

    return StravaTokenResponse(
        success=True,
        message=f"Strava account connected! Athlete: {athlete_name}",
    )


@router.post("/sync", response_model=dict)
def sync_strava_activities(
    days_back: int = Query(90, description="Number of days of activities to sync"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Sync recent Strava activities."""
    if not current_user.strava_access_token:
        raise HTTPException(status_code=400, detail="Strava not connected")

    client = StravaClient(current_user.strava_access_token)

    # Check token expiry and refresh if needed
    if current_user.strava_token_expires_at and current_user.strava_token_expires_at < datetime.utcnow().timestamp():
        token_data = client.refresh_token(current_user.strava_refresh_token)
        if token_data:
            current_user.strava_access_token = token_data["access_token"]
            current_user.strava_refresh_token = token_data["refresh_token"]
            current_user.strava_token_expires_at = token_data.get("expires_at", 0)
            db.commit()
            client = StravaClient(current_user.strava_access_token)
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

    existing_ids = set(
        row[0] for row in db.query(StravaActivity.strava_id).filter(
            StravaActivity.user_id == current_user.id
        ).all()
    )

    synced_count = 0
    ftp = current_user.ftp or 200

    for activity_data in all_activities:
        strava_id = activity_data["id"]
        if strava_id in existing_ids:
            continue

        activity_type = activity_data.get("type", "")
        if activity_type not in ("Ride", "VirtualRide", "Zwift"):
            continue

        activity = strava_activity_to_model(current_user.id, activity_data, ftp)
        db.add(activity)
        existing_ids.add(strava_id)
        synced_count += 1

    db.commit()

    return {
        "synced": synced_count,
        "total_strava": len(all_activities),
        "message": f"Synced {synced_count} new activities",
    }


@router.get("/activities", response_model=List[StravaActivityOut])
def list_strava_activities(
    limit: int = Query(50, le=200),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List synced Strava activities."""
    activities = (
        db.query(StravaActivity)
        .filter(StravaActivity.user_id == current_user.id)
        .order_by(desc(StravaActivity.start_date))
        .limit(limit)
        .all()
    )
    return activities
