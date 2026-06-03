"""Intervals.icu API client.

Full REST API client for Intervals.icu — a free training analytics platform
that provides deep cycling/running/triathlon metrics including CTL/ATL/TSB,
power curves, activity analysis, and training calendar management.

Docs: https://intervals.icu/api/v1/docs
Auth: HTTP Basic Auth with username='API_KEY' and password=<api_key>
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

    Uses HTTP Basic Auth with username='API_KEY' (not the athlete ID).
    All athlete-scoped endpoints are under /athlete/{athlete_id}/.

    Usage:
        client = IntervalsClient(api_key="...", athlete_id="i12345")
        athlete = await client.get_athlete()
        activities = await client.get_activities(days_back=90)
    """

    API_BASE = "https://intervals.icu/api/v1"
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
        self.athlete_id = athlete_id or settings.intervals_athlete_id

        # Base URL for athlete-scoped endpoints
        base = (base_url or settings.intervals_api_base or self.API_BASE).rstrip("/")
        if self.athlete_id:
            self.base_url = f"{base}/athlete/{self.athlete_id}"
        else:
            self.base_url = base

        # Auth: HTTP Basic Auth with username='API_KEY' (per OpenAPI spec)
        self._auth = httpx.BasicAuth("API_KEY", self.api_key) if self.api_key else None

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.TIMEOUT,
            auth=self._auth,
            headers=self._build_headers(),
        )

    @staticmethod
    def _build_headers() -> Dict[str, str]:
        """Build standard headers (auth is handled by BasicAuth)."""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "CycleTrain/1.0 (cycling-training-app)",
        }

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
            path: API endpoint path (relative to base URL)
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
        return await self._request("GET", "")

    async def update_athlete(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update athlete profile (FTP, weight, zones, etc.).

        Args:
            data: Fields to update

        Returns:
            Updated athlete profile
        """
        return await self._request("PUT", "", data=data)

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
            sport: Filter by sport type (Ride, Run, etc.)

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
        for each day via the athlete-summary endpoint.

        Args:
            days: Number of days of history

        Returns:
            List of daily metrics: {date, ctl, atl, tsb, total_tss, ...}
        """
        today = date.today()
        oldest = today - timedelta(days=days)
        params = {
            "oldest": oldest.isoformat(),
            "newest": today.isoformat(),
        }
        return await self._request("GET", "/athlete-summary", params=params)

    async def get_training_load(self) -> Dict[str, Any]:
        """Get current training load summary from recent athlete data.

        Returns:
            Dict with current_ctl, current_atl, current_tsb, etc.
        """
        metrics = await self.get_training_metrics(days=7)
        if metrics:
            latest = metrics[-1]
            return {
                "current_ctl": latest.get("fitness", 0),
                "current_atl": latest.get("fatigue", 0),
                "current_tsb": latest.get("form", 0),
            }
        return {"current_ctl": 0, "current_atl": 0, "current_tsb": 0}

    # ── Power Curve ──

    async def get_power_curve(
        self,
        days: int = 365,
        activity_type: str = "Ride",
    ) -> Dict[str, Any]:
        """Get power-duration curve data.

        Args:
            days: How far back to compute
            activity_type: Sport type (Ride, Run, etc.)

        Returns:
            Power curve data with durations and corresponding watts
        """
        params = {"type": activity_type}
        return await self._request("GET", "/power-curves", params=params)

    async def get_power_curve_trend(
        self,
        months: int = 12,
    ) -> List[Dict[str, Any]]:
        """Get power curve progression over time.

        Args:
            months: How many months of trend data

        Returns:
            List of monthly power curve snapshots
        """
        today = date.today()
        oldest = today - timedelta(days=months * 30)
        params = {
            "type": "Ride",
            "oldest": oldest.isoformat(),
            "newest": today.isoformat(),
        }
        return await self._request("GET", "/activity-power-curves", params=params)

    # ── Zones ──

    async def get_zones(self) -> Dict[str, Any]:
        """Get the athlete's power and HR zones from profile.

        Returns:
            Zone definitions for power, heart rate, and pace
        """
        athlete = await self.get_athlete()
        return {
            "power_zones": athlete.get("icu_power_zones"),
            "hr_zones": athlete.get("icu_hr_zones"),
            "ftp": athlete.get("icu_ftp"),
            "lthr": athlete.get("lthr"),
            "max_hr": athlete.get("athlete_max_hr"),
            "resting_hr": athlete.get("icu_resting_hr"),
            "weight": athlete.get("weight"),
        }

    async def update_zones(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update training zones via athlete profile update.

        Args:
            data: Zone definitions to update

        Returns:
            Updated athlete profile
        """
        return await self._request("PUT", "", data=data)

    # ── Events / Calendar ──

    async def get_events(
        self,
        start_date: date,
        end_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """Get calendar events (planned workouts, notes, etc.).

        Args:
            start_date: Start of date range
            end_date: End of date range (defaults to start_date + 7 days)

        Returns:
            List of event dicts
        """
        if not end_date:
            end_date = start_date + timedelta(days=7)
        params = {
            "oldest": start_date.isoformat(),
            "newest": end_date.isoformat(),
        }
        return await self._request("GET", "/events", params=params)

    # ── Webhooks (may not be under athlete prefix) ──

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
        raise NotImplementedError("Webhook API not available in current Intervals.icu API version")

    async def list_webhooks(self) -> List[Dict[str, Any]]:
        """List registered webhooks.

        Returns:
            List of webhook registrations
        """
        raise NotImplementedError("Webhook API not available in current Intervals.icu API version")

    async def delete_webhook(self, webhook_id: int):
        """Remove a webhook registration.

        Args:
            webhook_id: Webhook ID
        """
        raise NotImplementedError("Webhook API not available in current Intervals.icu API version")

    # ── Athlete Profile (detailed) ──

    async def get_profile(self) -> Dict[str, Any]:
        """Get extended athlete profile with shared folders and custom items."""
        return await self._request("GET", "/profile")


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
        "average_watts": activity.get("average_watts")
            or activity.get("avgPower")
            or activity.get("icu_average_watts"),
        "max_watts": activity.get("max_watts")
            or activity.get("maxPower")
            or activity.get("icu_max_watts"),
        "weighted_average_watts": activity.get("weighted_average_watts")
            or activity.get("normalizedPower")
            or activity.get("icu_weighted_avg_watts"),
        "average_heartrate": (
            activity.get("average_heartrate")
            or activity.get("avgHeartRate")
            or activity.get("average_heartrate", 0)
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

    Maps athlete-summary response fields to our app's expected format.

    Args:
        metrics: Raw training metrics dict from Intervals.icu

    Returns:
        Dict with our field names
    """
    return {
        "date": metrics.get("date"),
        "ctl": metrics.get("fitness", 0),
        "atl": metrics.get("fatigue", 0),
        "tsb": metrics.get("form", 0),
        "total_tss": metrics.get("training_load", 0),
        "total_duration_minutes": (
            (metrics.get("moving_time", 0) or metrics.get("time", 0)) / 60
        ),
        "total_distance_km": (
            metrics.get("distance", 0) / 1000
        ),
        "ride_count": metrics.get("count", 0),
    }
