"""Intervals.icu API client.

Full REST API client for Intervals.icu — a free training analytics platform
that provides deep cycling/running/triathlon metrics including CTL/ATL/TSB,
power curves, activity analysis, and training calendar management.

Docs: https://intervals.icu (API docs behind login)
Auth: API key (X-API-Key header) or OAuth 2.0
Base URL: https://intervals.icu/api/v1
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


# ── Exceptions ──


class IntervalsError(Exception):
    """Base exception for Intervals.icu API errors."""


class IntervalsAuthError(IntervalsError):
    """Authentication failed — check API key."""


class IntervalsRateLimitError(IntervalsError):
    """Rate limited by Intervals.icu API."""


class IntervalsNotFoundError(IntervalsError):
    """Requested resource not found."""


# ── Client ──


class IntervalsClient:
    """HTTP client for the Intervals.icu REST API.

    Supports both API key and OAuth 2.0 authentication.
    Provides methods for athletes, activities, training metrics,
    power curve, workouts, and webhooks.

    Usage:
        client = IntervalsClient(api_key="...")
        athlete = await client.get_athlete()
        activities = await client.get_activities(days_back=90)
    """

    BASE_URL = "https://intervals.icu/api/v1"
    TIMEOUT = 30

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        athlete_id: Optional[str] = None,
    ):
        """Initialise the client.

        Args:
            api_key: Intervals.icu API key. Falls back to settings.
            base_url: API base URL. Falls back to settings.
            athlete_id: Athlete ID for multi-athlete contexts.
        """
        self.api_key = api_key or settings.intervals_api_key
        self.base_url = (base_url or settings.intervals_api_base).rstrip("/")
        self.athlete_id = athlete_id or settings.intervals_athlete_id

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.TIMEOUT,
            headers=self._build_headers(),
        )

    def _build_headers(self) -> Dict[str, str]:
        """Build auth headers for the API client.

        Intervals.icu accepts:
        - API key via X-API-Key header
        - Or Bearer token via Authorization header

        Returns:
            Headers dict
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "CycleTrain/1.0 (cycling-training-app)",
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make an HTTP request to the Intervals.icu API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            path: API endpoint path (e.g. /athlete)
            params: Query parameters
            data: Request body for POST/PUT

        Returns:
            Parsed JSON response

        Raises:
            IntervalsAuthError: On 401
            IntervalsRateLimitError: On 429
            IntervalsNotFoundError: On 404
            IntervalsError: On other errors
        """
        url = f"{self.base_url}{path}"
        try:
            response = await self._client.request(
                method=method,
                url=url,
                params=params,
                json=data,
            )
        except httpx.TimeoutException:
            logger.error(f"Intervals.icu API timeout: {method} {path}")
            raise IntervalsError(f"Request timed out: {method} {path}")
        except httpx.RequestError as e:
            logger.error(f"Intervals.icu API request failed: {e}")
            raise IntervalsError(f"Request failed: {e}")

        if response.status_code == 401:
            raise IntervalsAuthError(
                "Invalid Intervals.icu API key. Check your settings."
            )
        elif response.status_code == 404:
            raise IntervalsNotFoundError(f"Resource not found: {path}")
        elif response.status_code == 429:
            raise IntervalsRateLimitError(
                "Intervals.icu rate limit exceeded. Try again later."
            )
        elif response.status_code >= 400:
            body = response.text[:200]
            raise IntervalsError(
                f"Intervals.icu API error {response.status_code}: {body}"
            )

        if response.status_code == 204:
            return None

        return response.json()

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    # ── Athlete ──

    async def get_athlete(self) -> Dict[str, Any]:
        """Get the authenticated athlete's profile.

        Returns:
            Athlete profile dict with name, FTP, weight, zones, etc.
        """
        return await self._request("GET", "/athlete")

    async def update_athlete(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update athlete profile (FTP, weight, zones, etc.).

        Args:
            data: Fields to update

        Returns:
            Updated athlete profile
        """
        return await self._request("PUT", "/athlete", data=data)

    # ── Activities ──

    async def get_activities(
        self,
        days_back: int = 90,
        limit: int = 100,
        offset: int = 0,
        sport: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent activities.

        Args:
            days_back: How many days of history to fetch
            limit: Max activities per page
            offset: Pagination offset
            sport: Filter by sport type (Cycling, Running, etc.)

        Returns:
            List of activity dicts with power, HR, duration, etc.
        """
        params: Dict[str, Any] = {
            "limit": min(limit, 200),
            "offset": offset,
        }
        if days_back:
            params["oldest"] = (date.today() - timedelta(days=days_back)).isoformat()
        if sport:
            params["sport"] = sport

        return await self._request("GET", "/activities", params=params)

    async def get_activity(self, activity_id: int) -> Dict[str, Any]:
        """Get detailed activity data including streams.

        Args:
            activity_id: Intervals.icu activity ID

        Returns:
            Activity detail dict with all metrics
        """
        return await self._request("GET", f"/activities/{activity_id}")

    async def get_activity_streams(
        self,
        activity_id: int,
        streams: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get time-series data streams for an activity.

        Args:
            activity_id: Intervals.icu activity ID
            streams: List of stream types (power, heartrate, cadence, etc.)

        Returns:
            Dict of stream name -> list of data points
        """
        params = {}
        if streams:
            params["streams"] = ",".join(streams)
        return await self._request(
            "GET", f"/activities/{activity_id}/streams", params=params
        )

    # ── Training Metrics (PMC) ──

    async def get_training_metrics(
        self,
        days: int = 90,
    ) -> List[Dict[str, Any]]:
        """Get Performance Management Chart data.

        Returns pre-computed CTL (fitness), ATL (fatigue), TSB (form)
        for each day.

        Args:
            days: Number of days of history

        Returns:
            List of daily metrics: {date, ctl, atl, tsb, total_tss, ...}
        """
        params = {"days": days}
        return await self._request("GET", "/training", params=params)

    async def get_training_load(self) -> Dict[str, Any]:
        """Get current training load summary.

        Returns current CTL, ATL, TSB with interpretations.

        Returns:
            Dict with current_ctl, current_atl, current_tsb, etc.
        """
        return await self._request("GET", "/training/load")

    # ── Power Curve ──

    async def get_power_curve(
        self,
        days: int = 365,
        model: str = "best",
    ) -> Dict[str, Any]:
        """Get power-duration curve data.

        Intervals.icu supports multiple models:
        - "best": Best actual efforts across all durations
        - "3p": Morton's 3-parameter critical power model
        - "monod": Monod & Scherrer model

        Args:
            days: How far back to compute
            model: Which model to use (best, 3p, monod)

        Returns:
            Power curve data with durations and corresponding watts
        """
        params = {"days": days, "model": model}
        return await self._request("GET", "/power-curve", params=params)

    async def get_power_curve_trend(
        self,
        months: int = 12,
    ) -> List[Dict[str, Any]]:
        """Get power curve progression over time (monthly snapshots).

        Args:
            months: How many months of trend data

        Returns:
            List of monthly power curve snapshots
        """
        params = {"months": months}
        return await self._request("GET", "/power-curve/trend", params=params)

    # ── Zones ──

    async def get_zones(self) -> Dict[str, Any]:
        """Get the athlete's power and HR zones.

        Returns:
            Zone definitions for power, heart rate, and pace
        """
        return await self._request("GET", "/zones")

    async def update_zones(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update training zones.

        Args:
            data: Zone definitions to update

        Returns:
            Updated zones
        """
        return await self._request("PUT", "/zones", data=data)

    # ── Workouts / Calendar ──

    async def get_workouts(
        self,
        start_date: date,
        end_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """Get planned workouts from the training calendar.

        Args:
            start_date: Start of date range
            end_date: End of date range (defaults to start_date + 7 days)

        Returns:
            List of workout dicts
        """
        if not end_date:
            end_date = start_date + timedelta(days=7)
        params = {
            "oldest": start_date.isoformat(),
            "newest": end_date.isoformat(),
        }
        return await self._request("GET", "/workouts", params=params)

    async def create_workout(self, workout: Dict[str, Any]) -> Dict[str, Any]:
        """Create a planned workout on the calendar.

        Args:
            workout: Workout dict with date, title, description, duration, etc.

        Returns:
            Created workout
        """
        return await self._request("POST", "/workouts", data=workout)

    async def update_workout(
        self,
        workout_id: int,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update an existing workout.

        Args:
            workout_id: Workout ID
            data: Fields to update

        Returns:
            Updated workout
        """
        return await self._request("PUT", f"/workouts/{workout_id}", data=data)

    async def delete_workout(self, workout_id: int):
        """Delete a workout from the calendar.

        Args:
            workout_id: Workout ID
        """
        await self._request("DELETE", f"/workouts/{workout_id}")

    # ── Wellbeing / Health ──

    async def get_wellbeing(
        self,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get wellbeing data (HRV, sleep, RHR, etc.).

        Args:
            days: How many days of data

        Returns:
            List of daily wellbeing records
        """
        params = {"days": days}
        return await self._request("GET", "/wellbeing", params=params)

    async def update_wellbeing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Log wellbeing data for today.

        Args:
            data: Wellbeing metrics (hrv, sleep_hours, resting_hr, etc.)

        Returns:
            Created/updated wellbeing record
        """
        return await self._request("POST", "/wellbeing", data=data)

    # ── Webhooks ──

    async def register_webhook(
        self,
        callback_url: str,
        events: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Register a webhook for real-time event notifications.

        Args:
            callback_url: URL to receive POST notifications
            events: Event types to subscribe to (default: all)

        Returns:
            Webhook registration details
        """
        data = {"callbackUrl": callback_url}
        if events:
            data["events"] = events
        return await self._request("POST", "/webhooks", data=data)

    async def list_webhooks(self) -> List[Dict[str, Any]]:
        """List registered webhooks.

        Returns:
            List of webhook registrations
        """
        return await self._request("GET", "/webhooks")

    async def delete_webhook(self, webhook_id: int):
        """Remove a webhook registration.

        Args:
            webhook_id: Webhook ID
        """
        await self._request("DELETE", f"/webhooks/{webhook_id}")


# ── Convenience ──


async def get_client() -> IntervalsClient:
    """Create and return an authenticated Intervals.icu client.

    Returns:
        Configured IntervalsClient instance

    Raises:
        IntervalsAuthError: If API key is not configured
    """
    if not settings.intervals_api_key:
        raise IntervalsAuthError(
            "Intervals.icu not configured. "
            "Set INTERVALS_API_KEY in your .env file."
        )
    return IntervalsClient()


def activity_to_dict(activity: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an Intervals.icu activity dict to our internal format.

    Maps Intervals.icu field names to our app's expected field names
    for compatibility with existing services (power curve, etc.).

    Args:
        activity: Raw activity dict from Intervals.icu API

    Returns:
        Dict with our internal field names
    """
    return {
        "id": activity.get("id"),
        "intervals_id": activity.get("id"),
        "name": activity.get("name", "Untitled"),
        "activity_type": activity.get("type", activity.get("sport", "Ride")),
        "start_date": activity.get("start_date") or activity.get("startDate"),
        "moving_time": activity.get("moving_time") or activity.get("movingTime", 0),
        "elapsed_time": activity.get("elapsed_time") or activity.get("elapsedTime", 0),
        "distance": activity.get("distance", 0),
        "total_elevation_gain": (
            activity.get("total_elevation_gain")
            or activity.get("elevationGain", 0)
        ),
        "average_watts": activity.get("average_watts") or activity.get("avgPower", 0),
        "max_watts": activity.get("max_watts") or activity.get("maxPower", 0),
        "weighted_average_watts": (
            activity.get("weighted_average_watts")
            or activity.get("normalizedPower", 0)
        ),
        "average_heartrate": (
            activity.get("average_heartrate")
            or activity.get("avgHeartRate", 0)
        ),
        "max_heartrate": activity.get("max_heartrate") or activity.get("maxHeartRate", 0),
        "average_cadence": activity.get("average_cadence") or activity.get("avgCadence", 0),
        "intensity_factor": activity.get("intensity_factor") or activity.get("intensityFactor", 0),
        "training_stress_score": activity.get("training_stress_score")
        or activity.get("tss", 0),
        "kilojoules": activity.get("kilojoules") or activity.get("kJ", 0),
        "perceived_exertion": activity.get("perceived_exertion") or activity.get("rpe"),
        "suffer_score": activity.get("suffer_score", 0),
        "training_load": activity.get("training_stress_score")
        or activity.get("tss", 0),
        "timezone": activity.get("timezone"),
    }


def training_metrics_to_dict(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Intervals.icu training metrics to our format.

    Args:
        metrics: Raw training metrics dict from Intervals.icu

    Returns:
        Dict with our field names
    """
    return {
        "date": metrics.get("date"),
        "ctl": metrics.get("ctl", metrics.get("fitness", 0)),
        "atl": metrics.get("atl", metrics.get("fatigue", 0)),
        "tsb": metrics.get("tsb", metrics.get("form", 0)),
        "total_tss": metrics.get("total_tss", metrics.get("tss", 0)),
        "total_duration_minutes": (
            metrics.get("total_duration_minutes")
            or (metrics.get("movingTime", 0) / 60)
            or 0
        ),
        "total_distance_km": (
            metrics.get("total_distance_km")
            or (metrics.get("distance", 0) / 1000)
            or 0
        ),
        "ride_count": metrics.get("ride_count", metrics.get("activityCount", 0)),
    }
