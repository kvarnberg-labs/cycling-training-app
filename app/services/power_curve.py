"""Power-duration curve analysis from Strava activity data.

Computes the athlete's best power output across standard time durations
(5s, 30s, 1min, 5min, 20min, 60min, 120min) by analyzing all synced
Strava activities. The resulting power curve shows the athlete's
power profile — a fundamental tool for identifying strengths and
weaknesses across different energy system durations.
"""

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.models import StravaActivity

logger = logging.getLogger(__name__)


# Standard durations for the power curve (in seconds)
POWER_CURVE_DURATIONS = [
    (5, "5s", "Peak Power"),
    (30, "30s", "Anaerobic"),
    (60, "1min", "VO2 Max"),
    (120, "2min", "Upper VO2"),
    (300, "5min", "VO2 Max / High Threshold"),
    (600, "10min", "Threshold"),
    (1200, "20min", "FTP / Threshold"),
    (1800, "30min", "Tempo / Threshold"),
    (3600, "60min", "Endurance / Tempo"),
    (7200, "2hr", "Endurance"),
]

# Duration bucket boundaries (in seconds): label, min_seconds, max_seconds
DURATION_BUCKETS = [
    ("5s", 0, 30, "max_watts"),
    ("30s", 30, 90, "weighted_average_watts"),
    ("1min", 90, 180, "weighted_average_watts"),
    ("2min", 180, 360, "weighted_average_watts"),
    ("5min", 360, 480, "weighted_average_watts"),
    ("10min", 480, 900, "weighted_average_watts"),
    ("20min", 900, 1800, "weighted_average_watts"),
    ("30min", 1800, 2700, "weighted_average_watts"),
    ("60min", 2700, 4500, "weighted_average_watts"),
    ("2hr", 4500, 999999, "weighted_average_watts"),
]


def compute_power_curve(
    db: Session,
    user_id: int,
    ftp: int = 200,
    days_back: int = 365,
) -> Dict[str, Any]:
    """Compute the athlete's power-duration curve from Strava activities.

    Analyzes all Strava activities within the specified time range and
    finds the best power output for each standard duration bucket.

    Args:
        db: Database session
        user_id: User ID
        ftp: User's FTP for reference
        days_back: How far back to look for activities

    Returns:
        Dict with:
            - curve: List of {label, duration_seconds, power_watts, percent_of_ftp, zone}
            - ftp: The FTP used for reference
            - estimated_ftp: Estimated FTP from the 20min best power * 0.95
            - best_20min_power: Best 20min power found
            - dominant_zone: Which energy system the athlete is strongest in
    """
    cutoff_date = date.today() - timedelta(days=days_back)

    activities = (
        db.query(StravaActivity)
        .filter(
            StravaActivity.user_id == user_id,
            StravaActivity.start_date >= cutoff_date,
            StravaActivity.average_watts.isnot(None),
        )
        .all()
    )

    if not activities:
        return {
            "curve": [],
            "ftp": ftp,
            "estimated_ftp": ftp,
            "best_20min_power": 0,
            "dominant_zone": "insufficient_data",
            "activity_count": 0,
        }

    # Compute best power for each duration bucket
    curve_points = _compute_best_efforts(activities, ftp)

    # Estimate FTP from 20min best power
    best_20min = 0
    for point in curve_points:
        if point["label"] == "20min":
            best_20min = point["power_watts"]
            break
    estimated_ftp = max(ftp, int(best_20min * 0.95)) if best_20min > 0 else ftp

    # Determine dominant zone
    dominant_zone = _determine_dominant_zone(curve_points)

    return {
        "curve": curve_points,
        "ftp": ftp,
        "estimated_ftp": estimated_ftp,
        "best_20min_power": best_20min,
        "dominant_zone": dominant_zone,
        "activity_count": len(activities),
    }


def _compute_best_efforts(
    activities: List[StravaActivity],
    ftp: int,
) -> List[Dict[str, Any]]:
    """Find the best power effort for each duration bucket.

    Args:
        activities: List of StravaActivity objects with power data
        ftp: User's FTP

    Returns:
        List of dicts with power curve points
    """
    best_powers: Dict[str, float] = {}
    best_activities: Dict[str, StravaActivity] = {}

    for act in activities:
        moving_time = act.moving_time or act.elapsed_time or 0
        if moving_time <= 0:
            continue

        # Find which bucket this activity falls into
        for label, min_sec, max_sec, power_field in DURATION_BUCKETS:
            if min_sec <= moving_time < max_sec:
                # Get the power value
                if power_field == "max_watts":
                    power = act.max_watts
                else:
                    power = act.weighted_average_watts or act.average_watts

                if power and power > 0:
                    if label not in best_powers or power > best_powers[label]:
                        best_powers[label] = power
                        best_activities[label] = act
                break

    # Ensure 5s bucket gets the absolute max_watts from any activity
    max_watts_overall = max((a.max_watts or 0) for a in activities)
    if max_watts_overall > 0:
        if "5s" not in best_powers or max_watts_overall > best_powers["5s"]:
            best_powers["5s"] = max_watts_overall

    # Build curve points in order
    curve_points = []
    for duration_s, label, zone_name in POWER_CURVE_DURATIONS:
        power = best_powers.get(label, 0)
        pct_ftp = round((power / ftp) * 100, 1) if ftp > 0 and power > 0 else 0

        # Interpret the zone
        if power == 0:
            zone = "no_data"
        elif pct_ftp >= 150:
            zone = "neuromuscular"
        elif pct_ftp >= 120:
            zone = "anaerobic"
        elif pct_ftp >= 105:
            zone = "vo2max"
        elif pct_ftp >= 90:
            zone = "threshold"
        elif pct_ftp >= 75:
            zone = "tempo"
        else:
            zone = "endurance"

        curve_points.append({
            "label": label,
            "duration_seconds": duration_s,
            "duration_label": zone_name,
            "power_watts": power,
            "percent_of_ftp": pct_ftp,
            "zone": zone,
        })

    return curve_points


def _determine_dominant_zone(
    curve_points: List[Dict[str, Any]],
) -> str:
    """Determine the athlete's strongest power zone.

    Compares actual power to expected power for each duration
    and identifies where the athlete exceeds expectations most.

    Args:
        curve_points: List of power curve data points

    Returns:
        String describing the dominant zone
    """
    # Expected power for each duration as % of FTP (typical values)
    expected_pct = {
        "5s": 180,
        "30s": 140,
        "1min": 120,
        "2min": 110,
        "5min": 105,
        "10min": 100,
        "20min": 95,
        "30min": 90,
        "60min": 85,
        "2hr": 80,
    }

    best_deviation = -999
    best_zone = "balanced"

    for point in curve_points:
        label = point["label"]
        expected = expected_pct.get(label)
        if expected and point["percent_of_ftp"] > 0:
            deviation = point["percent_of_ftp"] - expected
            if deviation > best_deviation:
                best_deviation = deviation
                best_zone = point["zone"]

    zone_labels = {
        "neuromuscular": "Sprint Power (Neuromuscular)",
        "anaerobic": "Short Power (Anaerobic)",
        "vo2max": "VO2 Max",
        "threshold": "Threshold Power",
        "tempo": "Tempo / Endurance",
        "endurance": "Aerobic Endurance",
        "no_data": "Insufficient Data",
    }

    return zone_labels.get(best_zone, "Balanced")


def get_power_curve_trend(
    db: Session,
    user_id: int,
    ftp: int = 200,
) -> List[Dict[str, Any]]:
    """Get power curve progression over time (monthly).

    Shows how the athlete's best 20min power has changed over recent months.

    Args:
        db: Database session
        user_id: User ID
        ftp: User's FTP

    Returns:
        List of monthly power curve snapshots
    """
    from collections import defaultdict

    activities = (
        db.query(StravaActivity)
        .filter(
            StravaActivity.user_id == user_id,
            StravaActivity.average_watts.isnot(None),
            StravaActivity.start_date >= date.today() - timedelta(days=365),
        )
        .order_by(StravaActivity.start_date)
        .all()
    )

    # Group by month
    monthly: Dict[str, List[float]] = defaultdict(list)
    for act in activities:
        if not act.start_date:
            continue
        month_key = act.start_date.strftime("%Y-%m")
        np = act.weighted_average_watts or act.average_watts or 0
        moving_time = act.moving_time or act.elapsed_time or 0
        # Only include longer activities (potential FTP estimation)
        if moving_time >= 1200 and np > 0:  # >= 20 min
            monthly[month_key].append(np)

    trend = []
    for month in sorted(monthly.keys()):
        np_values = monthly[month]
        if np_values:
            best_np = max(np_values)
            estimated_ftp = best_np * 0.95
            trend.append({
                "month": month,
                "best_20min_power": round(best_np, 0),
                "estimated_ftp": round(estimated_ftp, 0),
                "sample_count": len(np_values),
            })

    return trend
