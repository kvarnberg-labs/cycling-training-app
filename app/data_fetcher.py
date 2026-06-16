"""
Intervals.icu data fetcher — standalone module for pulling training data.

Designed for AI agent consumption. Fetches activities, PMC metrics,
and athlete profile from Intervals.icu with robust error handling.

Usage:
    from app.data_fetcher import TrainingDataFetcher
    fetcher = TrainingDataFetcher(api_key="...", athlete_id="i...")
    data = fetcher.fetch_all(days_back=30)

CLI:
    python -m app.data_fetcher --days 30
"""

import base64
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

API_BASE = "https://intervals.icu/api/v1"
TIMEOUT = 45

# ── Helpers ──


def _iso_today() -> str:
    return date.today().isoformat()


def _iso_days_back(days: int) -> str:
    return (date.today() - timedelta(days=days)).isoformat()


def _build_auth_header(api_key: str) -> Dict[str, str]:
    """Build HTTP Basic Auth header with username=API_KEY format."""
    creds = f"API_KEY:{api_key}"
    encoded = base64.b64encode(creds.encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def _make_request(
    method: str,
    url: str,
    api_key: str,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    """Synchronous HTTP request to Intervals.icu API."""
    headers = _build_auth_header(api_key)
    headers["Accept"] = "application/json"
    try:
        resp = httpx.request(
            method=method,
            url=url,
            params=params,
            headers=headers,
            timeout=TIMEOUT,
            follow_redirects=True,
        )
    except httpx.TimeoutException:
        raise RuntimeError(f"Intervals.icu API timeout: {method} {url}")
    except httpx.RequestError as e:
        raise RuntimeError(f"Intervals.icu request failed: {e}")

    if resp.status_code == 401:
        raise PermissionError("Invalid Intervals.icu API key — check your settings.")
    if resp.status_code == 403:
        raise PermissionError(f"Access denied — check athlete ID and key permissions. URL: {url}")
    if resp.status_code == 429:
        raise RuntimeError("Intervals.icu rate limit exceeded — try again later.")
    if resp.status_code >= 400:
        body = resp.text[:300]
        raise RuntimeError(f"Intervals.icu error {resp.status_code}: {body}")

    if resp.status_code == 204:
        return None
    return resp.json()


# ── Data mapping ──


def _activity_to_dict(act: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise an Intervals.icu activity into a clean output dict."""
    is_strava = act.get("source") == "STRAVA"
    strava_note = act.get("_note", "")

    # Extract PMC data that Intervals.icu sometimes includes per-activity
    fitness = act.get("icu_ctl") or 0
    fatigue = act.get("icu_atl") or 0
    form = act.get("form") or (fitness - fatigue if fitness and fatigue else 0)

    # Better naming for Strava-limited activities
    name = act.get("name")
    if not name and is_strava:
        name = f"Strava Activity ({act.get('start_date_local', '?')[:10]})"
    elif not name:
        name = "Untitled Activity"

    d = {
        "id": act.get("id"),
        "strava_id": act.get("strava_id"),
        "name": name,
        "activity_type": act.get("type") or act.get("sport") or "Ride",
        "start_date": act.get("start_date") or act.get("start_date_local"),
        "moving_time_seconds": act.get("moving_time") or 0,
        "elapsed_time_seconds": act.get("elapsed_time") or 0,
        "distance_meters": act.get("distance") or 0,
        "distance_km": round((act.get("distance") or 0) / 1000, 2),
        "elevation_gain": act.get("total_elevation_gain") or 0,
        "elevation_loss": act.get("total_elevation_loss") or 0,
        "average_watts": act.get("average_watts") or act.get("icu_average_watts"),
        "max_watts": act.get("max_watts") or act.get("icu_max_watts"),
        "weighted_avg_watts": act.get("weighted_average_watts") or act.get("icu_weighted_avg_watts"),
        "average_heartrate": act.get("average_heartrate") or act.get("average_heartrate"),
        "max_heartrate": act.get("max_heartrate"),
        "average_cadence": act.get("average_cadence"),
        "calories": act.get("calories") or act.get("kJ"),
        "intensity_factor": act.get("intensity_factor") or act.get("icu_intensity_factor"),
        "tss": act.get("training_stress_score") or act.get("icu_training_load"),
        "training_load": act.get("icu_training_load"),
        "workout_type": act.get("workout_type") or act.get("sub_type"),
        "commute": bool(act.get("commute")),
        "race": bool(act.get("race")),
        "trainer": bool(act.get("trainer")),
        "perceived_exertion": act.get("perceived_exertion"),
        "gear_id": act.get("gear", {}).get("id") if isinstance(act.get("gear"), dict) else None,
        "device_name": act.get("device_name"),
        "timezone": act.get("timezone"),
        "source": act.get("source"),
        "is_strava_limited": is_strava,
        "strava_note": strava_note if is_strava else None,
        # Per-activity PMC snapshot (intervals includes these per-activity)
        "fitness_ctl": fitness,
        "fatigue_atl": fatigue,
        "form_tsb": form,
        # Rolling FTP estimate
        "rolling_ftp": act.get("icu_rolling_ftp"),
    }

    # Clean up: remove None values for cleaner JSON
    return {k: v for k, v in d.items() if v is not None}


def _athlete_summary_to_dict(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise athlete-summary entry into PMC dict."""
    return {
        "date": metrics.get("date"),
        "fitness_ctl": round(metrics.get("fitness", 0), 1),
        "fatigue_atl": round(metrics.get("fatigue", 0), 1),
        "form_tsb": round(metrics.get("form", 0), 1),
        "total_tss": metrics.get("training_load", 0),
        "total_duration_minutes": (metrics.get("moving_time", 0) or metrics.get("time", 0)) / 60,
        "total_distance_km": round((metrics.get("distance", 0) or 0) / 1000, 1),
        "ride_count": metrics.get("count", 0),
        "elevation_gain": metrics.get("total_elevation_gain", 0),
        "calories": metrics.get("calories", 0),
    }


# ── Fetcher ──


class TrainingDataFetcher:
    """Pulls training data from Intervals.icu for AI consumption."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        athlete_id: Optional[str] = None,
        base_url: str = API_BASE,
    ):
        self.api_key = api_key or os.environ.get("INTERVALS_API_KEY", "")
        self.athlete_id = athlete_id or os.environ.get("INTERVALS_ATHLETE_ID", "")
        self.base_url = base_url.rstrip("/")

        if not self.api_key:
            raise ValueError("No Intervals.icu API key provided. Set INTERVALS_API_KEY env or pass api_key.")
        if not self.athlete_id:
            raise ValueError("No athlete ID provided. Set INTERVALS_ATHLETE_ID env or pass athlete_id.")

    def _url(self, path: str) -> str:
        """Build full URL for athlete-scoped endpoint."""
        return f"{self.base_url}/athlete/{self.athlete_id}{path}"

    # ── Public methods ──

    def get_athlete_profile(self) -> Dict[str, Any]:
        """Fetch athlete profile: FTP, weight, zones, etc."""
        data = _make_request("GET", self._url(""), self.api_key)
        sport_settings = data.get("sportSettings", [])
        ftp = None
        for ss in sport_settings:
            if ss.get("types") and "Ride" in ss.get("types", []):
                ftp = ss.get("ftp")
                break
        if ftp is None and sport_settings:
            ftp = sport_settings[0].get("ftp")

        return {
            "athlete_id": data.get("id"),
            "name": data.get("name"),
            "ftp": ftp,
            "weight_kg": data.get("icu_weight") or data.get("weight"),
            "resting_hr": data.get("icu_resting_hr"),
            "max_hr": data.get("athlete_max_hr"),
            "lthr": data.get("lthr"),
            "power_zones": data.get("icu_power_zones"),
            "hr_zones": data.get("icu_hr_zones"),
            "sweet_spot_min": data.get("icu_sweet_spot_min"),
            "sweet_spot_max": data.get("icu_sweet_spot_max"),
            "estimated_ftp": data.get("icu_rolling_ftp"),
            "time_zone": data.get("timezone"),
        }

    def get_activities(
        self,
        days_back: int = 90,
        limit: int = 200,
        offset: int = 0,
        sport: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch activities, filtering out limited STRAVA-only entries.

        Returns fully detailed activities. STRAVA-sourced activities are
        included but flagged with is_strava_limited=True and minimal fields.
        """
        params = {
            "limit": min(limit, 200),
            "offset": offset,
            "oldest": _iso_days_back(days_back),
        }
        if sport:
            params["sport"] = sport

        raw = _make_request("GET", self._url("/activities"), self.api_key, params=params)
        return [_activity_to_dict(a) for a in raw]

    def get_training_metrics(
        self,
        days_back: int = 42,
    ) -> List[Dict[str, Any]]:
        """Fetch PMC data (CTL/ATL/TSB) from athlete-summary.

        The Intervals.icu athlete-summary endpoint returns weekly rollups.
        We pull a wider window and extract daily PMC data.
        """
        today = date.today()
        oldest = today - timedelta(days=days_back)
        params = {"oldest": oldest.isoformat(), "newest": today.isoformat()}
        raw = _make_request("GET", self._url("/athlete-summary"), self.api_key, params=params)

        if not raw:
            return []

        # athlete-summary returns list of weekly rollups — extract PMC from the
        # first (most recent) entry since it represents up-to-date metrics
        metrics = []
        for entry in raw:
            metrics.append(_athlete_summary_to_dict(entry))
        return metrics

    def get_power_curves(self, days: int = 365, activity_type: str = "Ride") -> Dict[str, Any]:
        """Fetch power-duration curve data."""
        params = {"type": activity_type}
        return _make_request("GET", self._url("/power-curves"), self.api_key, params=params)

    def fetch_all(self, days_back: int = 42) -> Dict[str, Any]:
        """Fetch everything in one call — profile, activities, PMC.

        Returns a single structured dict ready for LLM consumption.
        """
        profile = self.get_athlete_profile()
        activities = self.get_activities(days_back=days_back)
        pmc = self.get_training_metrics(days_back=days_back)

        # Calculate weekly summary
        today = date.today()
        week_ago = today - timedelta(days=7)
        week_activities = [a for a in activities if a.get("start_date", "")[:10] >= week_ago.isoformat()]

        # Summary stats for recent week
        weekly_summary = {
            "ride_count": len([a for a in week_activities if a.get("activity_type") in ("Ride", "VirtualRide", "Zwift")]),
            "run_count": len([a for a in week_activities if a.get("activity_type") == "Run"]),
            "total_distance_km": round(sum(a.get("distance_km", 0) for a in week_activities), 1),
            "total_tss": sum(a.get("tss", 0) or 0 for a in week_activities),
            "total_time_minutes": sum(a.get("moving_time_seconds", 0) for a in week_activities) // 60,
            "total_elevation_gain": sum(a.get("elevation_gain", 0) for a in week_activities),
        }

        return {
            "fetched_at": datetime.now().isoformat(),
            "athlete": profile,
            "training_overview": {
                "days_back": days_back,
                "total_activities": len(activities),
                "strava_limited_count": len([a for a in activities if a.get("is_strava_limited")]),
            },
            "weekly_summary": weekly_summary,
            "pmc": pmc,
            "activities": activities,
        }


# ── CLI entrypoint ──


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch training data from Intervals.icu")
    parser.add_argument("--days", type=int, default=42, help="Days of history to fetch")
    parser.add_argument("--api-key", help="Intervals.icu API key (default: INTERVALS_API_KEY env)")
    parser.add_argument("--athlete-id", help="Intervals.icu athlete ID (default: INTERVALS_ATHLETE_ID env)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--activities-only", action="store_true", help="Only fetch activities")
    parser.add_argument("--pmc-only", action="store_true", help="Only fetch PMC data")
    parser.add_argument("--compact", action="store_true",
                        help="Compact mode: omit full activity details, just summary")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    try:
        fetcher = TrainingDataFetcher(api_key=args.api_key, athlete_id=args.athlete_id)

        if args.activities_only:
            activities = fetcher.get_activities(days_back=args.days)
            data = {"activities": activities, "count": len(activities)}
        elif args.pmc_only:
            pmc = fetcher.get_training_metrics(days_back=args.days)
            profile = fetcher.get_athlete_profile()
            data = {"athlete": profile, "pmc": pmc}
        elif args.compact:
            full = fetcher.fetch_all(days_back=args.days)
            # Keep summary, drop full activity list
            full.pop("activities", None)
            data = full
        else:
            data = fetcher.fetch_all(days_back=args.days)

        indent = 2 if args.pretty else None
        print(json.dumps(data, indent=indent, default=str))
    except (ValueError, PermissionError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
