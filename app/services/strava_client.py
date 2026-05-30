"""Strava API v3 client for fetching athlete activities."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import json
import logging

import httpx

from app.config import settings
from app.models import StravaActivity, WorkoutType, TrainingGoal

logger = logging.getLogger(__name__)

STRAVA_API_BASE = "https://www.strava.com/api/v3"
STRAVA_AUTH_BASE = "https://www.strava.com/oauth"


class StravaClient:
    """Client for the Strava v3 REST API."""

    def __init__(self, access_token: str):
        self.access_token = access_token
        self.client = httpx.Client(
            base_url=STRAVA_API_BASE,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30.0,
        )

    def refresh_token(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        """Refresh the Strava access token.

        Args:
            refresh_token: The Strava refresh token

        Returns:
            Dict with new tokens if successful, None otherwise
        """
        try:
            resp = httpx.post(
                f"{STRAVA_AUTH_BASE}/token",
                data={
                    "client_id": settings.strava_client_id,
                    "client_secret": settings.strava_client_secret,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            # Update our internal client
            self.access_token = data["access_token"]
            self.client.headers["Authorization"] = f"Bearer {self.access_token}"
            return data
        except Exception as e:
            logger.error(f"Failed to refresh Strava token: {e}")
            return None

    def get_athlete(self) -> Optional[Dict[str, Any]]:
        """Get the authenticated athlete's profile.

        Returns:
            Athlete info dict or None on failure.
        """
        try:
            resp = self.client.get("/athlete")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to get athlete: {e}")
            return None

    def get_activities(
        self,
        page: int = 1,
        per_page: int = 30,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch the athlete's activities.

        Args:
            page: Page number (1-indexed)
            per_page: Activities per page (max 200)
            after: Only activities after this timestamp
            before: Only activities before this timestamp

        Returns:
            List of activity dicts
        """
        params: Dict[str, Any] = {
            "page": page,
            "per_page": min(per_page, 200),
        }
        if after:
            params["after"] = int(after.timestamp())
        if before:
            params["before"] = int(before.timestamp())

        try:
            resp = self.client.get("/athlete/activities", params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch activities (page {page}): {e}")
            return []

    def get_activity_by_id(self, activity_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a single activity (includes streams).

        Args:
            activity_id: Strava activity ID

        Returns:
            Activity detail dict or None
        """
        try:
            resp = self.client.get(f"/activities/{activity_id}", params={"include_all_efforts": True})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to get activity {activity_id}: {e}")
            return None

    def get_activity_streams(
        self, activity_id: int, keys: str = "time,distance,heartrate,watts,cadence,velocity_smooth,altitude"
    ) -> Optional[Dict[str, Any]]:
        """Get activity streams (time-series data).

        Args:
            activity_id: Strava activity ID
            keys: Comma-separated stream types

        Returns:
            Stream data dict or None
        """
        try:
            resp = self.client.get(
                f"/activities/{activity_id}/streams",
                params={"keys": keys, "key_by_type": "true"},
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to get streams for {activity_id}: {e}")
            return None


def compute_tss_from_activity(
    activity_data: Dict[str, Any],
    ftp: int,
) -> tuple:
    """Compute training load metrics from a Strava activity dict.

    Args:
        activity_data: Raw activity data from Strava API
        ftp: User's FTP

    Returns:
        Tuple of (tss, normalized_power, intensity_factor, workout_type, training_load)
    """
    from app.services.training_load import calculate_tss, classify_workout_type

    # Get power data
    weighted_avg_watts = activity_data.get("weighted_average_watts")
    average_watts = activity_data.get("average_watts")
    max_watts = activity_data.get("max_watts")
    moving_time = activity_data.get("moving_time", 0) or 0
    elapsed_time = activity_data.get("elapsed_time", 0) or 0

    # Use NP if available, fall back to average power, then None
    np = weighted_avg_watts or average_watts or 0
    avg_hr = activity_data.get("average_heartrate")
    max_hr = activity_data.get("max_heartrate")

    duration = moving_time if moving_time > 0 else elapsed_time

    tss = 0.0
    intensity_factor = 0.0
    workout_type = None

    if np > 0 and ftp > 0:
        tss = calculate_tss(np, ftp, duration)
        intensity_factor = np / ftp
        workout_type = classify_workout_type(np, ftp, avg_hr, max_hr, None)
    elif avg_hr:
        # Estimate from HR if no power
        from app.config import settings as s
        from app.services.training_load import estimate_tss_from_hr
        tss = estimate_tss_from_hr(
            avg_hr, s.default_hr_rest, s.default_hr_max,
            duration / 60, ftp, s.default_weight
        )
        intensity_factor = (avg_hr - s.default_hr_rest) / (s.default_hr_max - s.default_hr_rest) * 1.0
        workout_type = classify_workout_type(ftp * intensity_factor, ftp, avg_hr, max_hr, None)

    training_load = tss  # Use TSS as the primary load metric
    tss = round(tss, 1) if tss else 0.0

    return tss, np, round(intensity_factor, 3) if intensity_factor else 0.0, workout_type, training_load


def strava_activity_to_model(
    user_id: int,
    activity_data: Dict[str, Any],
    ftp: int,
) -> StravaActivity:
    """Convert a Strava API activity dict to a SQLAlchemy model instance.

    Args:
        user_id: Internal user ID
        activity_data: Activity dict from Strava API
        ftp: User's FTP

    Returns:
        StravaActivity model instance (not yet committed)
    """
    tss, np, if_ratio, workout_type, training_load = compute_tss_from_activity(activity_data, ftp)
    w_type = None
    if workout_type:
        try:
            w_type = WorkoutType(workout_type)
        except ValueError:
            w_type = None

    start_date_str = activity_data.get("start_date")
    start_date = None
    if start_date_str:
        try:
            start_date = datetime.fromisoformat(start_date_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            start_date = datetime.utcnow()

    return StravaActivity(
        user_id=user_id,
        strava_id=activity_data["id"],
        name=activity_data.get("name"),
        activity_type=activity_data.get("type", "Ride"),
        start_date=start_date,
        timezone=activity_data.get("timezone"),
        elapsed_time=activity_data.get("elapsed_time"),
        moving_time=activity_data.get("moving_time"),
        distance=activity_data.get("distance"),
        total_elevation_gain=activity_data.get("total_elevation_gain"),
        average_watts=activity_data.get("average_watts"),
        max_watts=activity_data.get("max_watts"),
        weighted_average_watts=np,
        average_heartrate=activity_data.get("average_heartrate"),
        max_heartrate=activity_data.get("max_heartrate"),
        average_cadence=activity_data.get("average_cadence"),
        kilojoules=activity_data.get("kilojoules"),
        suffer_score=activity_data.get("suffer_score"),
        training_load=training_load,
        workout_type=w_type,
        intensity_factor=if_ratio,
        training_stress_score=tss,
        raw_data=json.dumps(activity_data, default=str),
    )


def get_strava_oauth_url() -> str:
    """Generate the Strava OAuth authorization URL.

    Returns:
        Full authorization URL string
    """
    params = (
        f"client_id={settings.strava_client_id}"
        f"&redirect_uri={settings.strava_redirect_uri}"
        f"&response_type=code"
        f"&approval_prompt=auto"
        f"&scope=read,activity:read_all,profile:read_all"
    )
    return f"{STRAVA_AUTH_BASE}/authorize?{params}"


def exchange_authorization_code(code: str) -> Optional[Dict[str, Any]]:
    """Exchange an OAuth authorization code for tokens.

    Args:
        code: The authorization code from Strava redirect

    Returns:
        Dict with tokens and athlete info, or None on failure
    """
    try:
        resp = httpx.post(
            f"{STRAVA_AUTH_BASE}/token",
            data={
                "client_id": settings.strava_client_id,
                "client_secret": settings.strava_client_secret,
                "code": code,
                "grant_type": "authorization_code",
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"Failed to exchange auth code: {e}")
        return None
