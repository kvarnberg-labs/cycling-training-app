"""Weather service for fetching forecasts and determining indoor/outdoor suitability.

Uses SMHI API (free, no key required, excellent Nordic coverage) as the primary
source, with OpenWeatherMap as fallback if configured.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# ── Weather condition categories ──

# SMHI weather symbols (param 17 = wsymb, 18 = wsymb2)
# https://opendata.smhi.se/apidocs/metfcst/parameters.html
SMHI_SYMBOLS: Dict[int, str] = {
    1: "clear",
    2: "nearly_clear",
    3: "partly_cloudy",
    4: "cloudy",
    5: "very_cloudy",
    6: "overcast",
    7: "fog",
    8: "rain_light",
    9: "rain_moderate",
    10: "rain_heavy",
    11: "thunder",
    12: "thunder_rain",
    13: "sleet_light",
    14: "sleet_moderate",
    15: "sleet_heavy",
    16: "snow_light",
    17: "snow_moderate",
    18: "snow_heavy",
    19: "rain_thunder",
    20: "snow_thunder",
    21: "sleet_thunder",
}

# Weather conditions that suggest indoor training
BAD_WEATHER_SYMBOLS = {
    # Rain
    "rain_light", "rain_moderate", "rain_heavy",
    # Sleet
    "sleet_light", "sleet_moderate", "sleet_heavy",
    # Snow
    "snow_light", "snow_moderate", "snow_heavy",
    # Thunderstorms
    "thunder", "thunder_rain", "rain_thunder", "snow_thunder", "sleet_thunder",
}

# Weather conditions that suggest outdoor training (layer up)
MARGINAL_WEATHER_SYMBOLS = {
    "fog",
    "overcast",
    "very_cloudy",
}

# Weather conditions good for outdoor riding
GOOD_WEATHER_SYMBOLS = {
    "clear",
    "nearly_clear",
    "partly_cloudy",
    "cloudy",
}

# ── Icons ──

WEATHER_ICONS: Dict[str, str] = {
    "clear": "☀️",
    "nearly_clear": "🌤️",
    "partly_cloudy": "⛅",
    "cloudy": "☁️",
    "very_cloudy": "☁️",
    "overcast": "☁️",
    "fog": "🌫️",
    "rain_light": "🌦️",
    "rain_moderate": "🌧️",
    "rain_heavy": "🌧️",
    "thunder": "⛈️",
    "thunder_rain": "⛈️",
    "rain_thunder": "⛈️",
    "snow_thunder": "⛈️",
    "sleet_thunder": "⛈️",
    "sleet_light": "🌨️",
    "sleet_moderate": "🌨️",
    "sleet_heavy": "🌨️",
    "snow_light": "❄️",
    "snow_moderate": "❄️",
    "snow_heavy": "❄️",
}

WEATHER_CONDITION_LABELS: Dict[str, str] = {
    "clear": "Clear",
    "nearly_clear": "Nearly Clear",
    "partly_cloudy": "Partly Cloudy",
    "cloudy": "Cloudy",
    "very_cloudy": "Very Cloudy",
    "overcast": "Overcast",
    "fog": "Fog",
    "rain_light": "Light Rain",
    "rain_moderate": "Moderate Rain",
    "rain_heavy": "Heavy Rain",
    "thunder": "Thunder",
    "thunder_rain": "Thunder & Rain",
    "rain_thunder": "Rain & Thunder",
    "snow_thunder": "Snow & Thunder",
    "sleet_thunder": "Sleet & Thunder",
    "sleet_light": "Light Sleet",
    "sleet_moderate": "Moderate Sleet",
    "sleet_heavy": "Heavy Sleet",
    "snow_light": "Light Snow",
    "snow_moderate": "Moderate Snow",
    "snow_heavy": "Heavy Snow",
}


@dataclass
class WeatherForecast:
    """Weather forecast for a single day."""
    date: date
    symbol: str  # e.g. "clear", "rain_moderate"
    icon: str    # Emoji icon
    label: str   # Human-readable label
    temp_min: float
    temp_max: float
    precipitation_mm: float
    wind_speed_ms: float
    is_indoor_suitable: bool  # Bad weather → indoor
    is_outdoor_suitable: bool  # Good enough for outdoor riding


def _smhi_api_url(lat: float, lon: float) -> str:
    """Build SMHI API URL for a given lat/lon."""
    return (
        f"https://opendata-download-metfcst.smhi.se/api/category/pmp3g/version/2/"
        f"geotype/point/lon/{lon}/lat/{lat}/data.json"
    )


async def _fetch_smhi_forecast(lat: float, lon: float) -> Optional[dict]:
    """Fetch SMHI forecast data for a location.

    SMHI provides 10-day forecast data with 3-hourly resolution.
    Returns the raw JSON response or None on failure.
    """
    url = _smhi_api_url(lat, lon)
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        logger.warning(f"SMHI API error for {lat},{lon}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching SMHI forecast: {e}")
        return None


def _parse_smhi_forecast(data: dict) -> List[WeatherForecast]:
    """Parse SMHI forecast JSON into daily WeatherForecast objects.

    SMHI response format:
    {
      "approvedTime": "...",
      "referenceTime": "...",
      "geometry": { "coordinates": [[lon, lat]] },
      "timeSeries": [
        {
          "validTime": "2026-05-31T12:00:00Z",
          "parameters": [
            {"name": "t", "values": [15.2]},        # Temperature (C)
            {"name": "ws", "values": [4.5]},         # Wind speed (m/s)
            {"name": "pmin", "values": [0.2]},       # Precipitation min (mm)
            {"name": "pmax", "values": [0.5]},       # Precipitation max (mm)
            {"name": "Wsymb2", "values": [3]},       # Weather symbol (1-21)
            ...
          ]
        },
        ...
      ]
    }
    """
    daily_data: Dict[date, dict] = {}

    for entry in data.get("timeSeries", []):
        valid_time = entry.get("validTime", "")
        try:
            # Parse ISO datetime like "2026-05-31T12:00:00Z"
            dt = datetime.fromisoformat(valid_time.replace("Z", "+00:00"))
            day = dt.date()
        except (ValueError, AttributeError):
            continue

        if day not in daily_data:
            daily_data[day] = {
                "symbols": [],
                "temps": [],
                "precip_min": 0.0,
                "wind_speeds": [],
            }

        params = {p["name"]: p["values"] for p in entry.get("parameters", [])}

        if "t" in params:
            daily_data[day]["temps"].append(params["t"][0])
        if "ws" in params:
            daily_data[day]["wind_speeds"].append(params["ws"][0])
        if "pmin" in params:
            daily_data[day]["precip_min"] += params["pmin"][0]
        if "Wsymb2" in params:
            daily_data[day]["symbols"].append(params["Wsymb2"][0])

    forecasts = []
    now = datetime.utcnow().date()

    for day, data in daily_data.items():
        # Only include the next 7 days
        if day < now or day > now + timedelta(days=7):
            continue

        symbols = data.get("symbols", [])
        # Pick the most common symbol during daytime hours (6am-6pm)
        # For simplicity, take the most common symbol overall
        if symbols:
            symbol_code = max(set(symbols), key=symbols.count)
            symbol = SMHI_SYMBOLS.get(symbol_code, "cloudy")
        else:
            symbol = "cloudy"

        temps = data.get("temps", [])
        temp_min = min(temps) if temps else 0.0
        temp_max = max(temps) if temps else 0.0

        wind_speeds = data.get("wind_speeds", [])
        wind_speed_ms = max(wind_speeds) if wind_speeds else 0.0

        precipitation_mm = data.get("precip_min", 0.0)

        is_indoor = symbol in BAD_WEATHER_SYMBOLS
        is_outdoor = symbol in GOOD_WEATHER_SYMBOLS or (
            symbol in MARGINAL_WEATHER_SYMBOLS and precipitation_mm < 1.0 and wind_speed_ms < 8.0
        )

        forecasts.append(WeatherForecast(
            date=day,
            symbol=symbol,
            icon=WEATHER_ICONS.get(symbol, "❓"),
            label=WEATHER_CONDITION_LABELS.get(symbol, symbol),
            temp_min=round(temp_min, 1),
            temp_max=round(temp_max, 1),
            precipitation_mm=round(precipitation_mm, 1),
            wind_speed_ms=round(wind_speed_ms, 1),
            is_indoor_suitable=is_indoor,
            is_outdoor_suitable=is_outdoor,
        ))

    return forecasts


async def get_forecast(lat: float, lon: float) -> List[WeatherForecast]:
    """Get weather forecast for a location.

    Tries SMHI first (free, no API key required), returns up to 7 days.

    Args:
        lat: Latitude (e.g. 59.33 for Stockholm)
        lon: Longitude (e.g. 18.07 for Stockholm)

    Returns:
        List of WeatherForecast objects, one per day.
        Empty list if no data available.
    """
    data = await _fetch_smhi_forecast(lat, lon)
    if data:
        forecasts = _parse_smhi_forecast(data)
        if forecasts:
            return forecasts

    logger.warning(f"No forecast data available for {lat},{lon}")
    return []


def get_weather_recommendation(
    symbol: str,
    preference: str = "auto",
) -> Dict:
    """Determine training recommendation based on weather and user preference.

    Args:
        symbol: Weather symbol string (e.g. "clear", "rain_moderate")
        preference: User preference ("auto", "indoor", "outdoor")

    Returns:
        Dict with recommendation info:
            - indoor: bool — whether indoor training is recommended
            - suggestion: str — human-readable suggestion
            - reason: str — why this suggestion was made
    """
    if preference == "indoor":
        return {
            "indoor": True,
            "suggestion": "indoor_workout",
            "reason": "User preference set to indoor",
        }

    if preference == "outdoor":
        return {
            "indoor": False,
            "suggestion": "outdoor_endurance",
            "reason": "User preference set to outdoor",
        }

    # Auto mode — base on weather
    if symbol in BAD_WEATHER_SYMBOLS:
        label = WEATHER_CONDITION_LABELS.get(symbol, "bad weather")
        return {
            "indoor": True,
            "suggestion": "indoor_workout",
            "reason": f"Weather is {label.lower()} — recommend indoor/Zwift workout",
        }

    if symbol in MARGINAL_WEATHER_SYMBOLS:
        label = WEATHER_CONDITION_LABELS.get(symbol, "marginal")
        return {
            "indoor": False,
            "suggestion": "outdoor_endurance",
            "reason": f"Weather is {label.lower()} but rideable with proper gear",
        }

    if symbol in GOOD_WEATHER_SYMBOLS:
        label = WEATHER_CONDITION_LABELS.get(symbol, "good")
        return {
            "indoor": False,
            "suggestion": "outdoor_endurance",
            "reason": f"Weather is {label.lower()} — great for outdoor riding",
        }

    # Unknown condition — default to outdoor
    return {
        "indoor": False,
        "suggestion": "outdoor_endurance",
        "reason": "Weather condition unknown — defaulting to outdoor",
    }
