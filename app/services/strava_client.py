"""Strava activity conversion helpers.

Contains data processing functions that convert raw activity data (from
MCP tools or REST API) into the internal StravaActivity model.

This module NO LONGER contains a REST API client — all Strava API
interaction now goes through the MCP client (strava_mcp_client.py).
"""

from datetime import datetime
from typing import Any, Dict, Optional
import json
import logging

from app.models import StravaActivity, WorkoutType

logger = logging.getLogger(__name__)


def compute_tss_from_activity(
    activity_data: Dict[str, Any],
    ftp: int,
) -> tuple:
    """Compute training load metrics from a Strava activity dict.

    Args:
        activity_data: Raw activity data from Strava API/MCP
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

    training_load = tss
    tss = round(tss, 1) if tss else 0.0

    return tss, np, round(intensity_factor, 3) if intensity_factor else 0.0, workout_type, training_load


def strava_activity_to_model(
    user_id: int,
    activity_data: Dict[str, Any],
    ftp: int,
) -> StravaActivity:
    """Convert a Strava activity dict to a SQLAlchemy model instance.

    Args:
        user_id: Internal user ID
        activity_data: Activity dict (from MCP tools or REST API)
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
        strava_id=activity_data.get("id") or activity_data.get("strava_id", 0),
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
