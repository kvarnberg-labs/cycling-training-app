"""Strava MCP-based router — OAuth and activity sync via MCP tools.

Replaces the old REST-API-based router. Communicates with Strava through
the @r-huijts/strava-mcp-server via the Model Context Protocol.
"""

from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import get_db
from app.models import User, StravaActivity, Workout, WorkoutStatus
from app.schemas import StravaAuthUrl, StravaTokenResponse, StravaActivityOut
from app.auth import get_current_user
from app.services.strava_mcp_client import (
    fetch_all_recent_activities,
    check_connection,
    connect_strava as mcp_connect_strava,
    disconnect_strava as mcp_disconnect_strava,
)
from app.services.strava_client import strava_activity_to_model

router = APIRouter(prefix="/strava", tags=["strava"])


@router.get("/auth-url", response_model=StravaAuthUrl)
async def get_auth_url():
    """Get the Strava MCP connection URL.

    The MCP server handles OAuth internally. This endpoint triggers
    the MCP server's browser-based OAuth flow which manages its own
    token storage at ~/.config/strava-mcp/config.json.
    """
    return StravaAuthUrl(auth_url="/api/strava/mcp-connect")


@router.post("/mcp-connect")
async def mcp_connect(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Initiate Strava connection via MCP server.

    The MCP server handles the full OAuth flow (opens a browser window
    for the user to authorize). Tokens are stored locally by the MCP server.
    """
    result = await mcp_connect_strava()
    if result.get("error"):
        raise HTTPException(status_code=400, detail=f"Strava MCP connection failed: {result['error']}")

    # Mark user as Strava-connected in the app
    current_user.strava_athlete_id = 1  # Placeholder — MCP manages its own auth
    current_user.strava_access_token = "__mcp_managed__"
    db.commit()

    return StravaTokenResponse(
        success=True,
        message="Strava account connected via MCP! You can now sync activities.",
    )


@router.post("/sync", response_model=dict)
async def sync_strava_activities(
    days_back: int = Query(90, description="Number of days of activities to sync"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Sync recent Strava activities via MCP tools."""
    # Check connection via MCP
    connected = await check_connection()
    if not connected:
        raise HTTPException(
            status_code=400,
            detail="Strava not connected. Use the Strava MCP connection flow first.",
        )

    # Fetch activities via MCP
    after_date = datetime.utcnow() - timedelta(days=days_back)
    all_activities = await fetch_all_recent_activities(
        after=after_date,
        activity_types=["Ride", "VirtualRide", "Zwift"],
    )

    if not all_activities:
        return {"synced": 0, "message": "No new activities found"}

    # Deduplicate against existing
    existing_ids = set(
        row[0] for row in db.query(StravaActivity.strava_id).filter(
            StravaActivity.user_id == current_user.id
        ).all()
    )

    synced_count = 0
    ftp = current_user.ftp or 200

    for activity_data in all_activities:
        strava_id = activity_data.get("id") or activity_data.get("strava_id")
        if not strava_id or strava_id in existing_ids:
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
    """List synced Strava activities from the local database."""
    activities = (
        db.query(StravaActivity)
        .filter(StravaActivity.user_id == current_user.id)
        .order_by(desc(StravaActivity.start_date))
        .limit(limit)
        .all()
    )
    return activities


@router.get("/connection-status")
async def connection_status(
    current_user: User = Depends(get_current_user),
):
    """Check Strava MCP connection status."""
    connected = await check_connection()
    return {
        "connected": connected,
        "method": "mcp",
        "server": "@r-huijts/strava-mcp-server",
    }
