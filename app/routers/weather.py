"""Weather router — forecast and recommendations."""

from datetime import date, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.schemas import WeatherDayOut, WeatherForecastResponse
from app.services.weather import get_forecast, get_weather_recommendation, WeatherForecast
from app.auth import get_current_user

router = APIRouter(prefix="/weather", tags=["weather"])


@router.get("/forecast", response_model=WeatherForecastResponse)
async def forecast(
    lat: Optional[float] = Query(None, description="Latitude (defaults to user's saved location)"),
    lon: Optional[float] = Query(None, description="Longitude (defaults to user's saved location)"),
    days: int = Query(7, le=7, description="Number of forecast days"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get weather forecast for the user's location.

    Uses the user's saved location (lat/lon) from settings, or the provided
    lat/lon query parameters.
    """
    # Determine location: prefer explicit params, fall back to user settings
    use_lat = lat if lat is not None else current_user.location_lat
    use_lon = lon if lon is not None else current_user.location_lon

    if use_lat is None or use_lon is None:
        raise HTTPException(
            status_code=400,
            detail="No location configured. Set your location in Settings, or provide lat/lon parameters.",
        )

    forecasts = await get_forecast(use_lat, use_lon)

    return WeatherForecastResponse(
        location_lat=use_lat,
        location_lon=use_lon,
        days=[_forecast_to_day(f) for f in forecasts[:days]],
    )


@router.get("/recommendation", response_model=Dict)
async def weather_recommendation(
    date_str: Optional[str] = Query(None, description="Date (YYYY-MM-DD), defaults to today"),
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
    current_user: User = Depends(get_current_user),
):
    """Get a single-day weather-based training recommendation."""
    target_date = date.today()
    if date_str:
        try:
            target_date = date.fromisoformat(date_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    use_lat = lat if lat is not None else current_user.location_lat
    use_lon = lon if lon is not None else current_user.location_lon

    if use_lat is None or use_lon is None:
        raise HTTPException(
            status_code=400,
            detail="No location configured. Set your location in Settings, or provide lat/lon parameters.",
        )

    forecasts = await get_forecast(use_lat, use_lon)
    target_forecast = next((f for f in forecasts if f.date == target_date), None)

    if not target_forecast:
        return {
            "date": target_date.isoformat(),
            "available": False,
            "message": "No forecast available for this date.",
        }

    recommendation = get_weather_recommendation(
        target_forecast.symbol,
        preference=current_user.weather_preference or "auto",
    )

    return {
        "date": target_date.isoformat(),
        "available": True,
        "symbol": target_forecast.symbol,
        "icon": target_forecast.icon,
        "label": target_forecast.label,
        "temp_min": target_forecast.temp_min,
        "temp_max": target_forecast.temp_max,
        "precipitation_mm": target_forecast.precipitation_mm,
        "wind_speed_ms": target_forecast.wind_speed_ms,
        "indoor": recommendation["indoor"],
        "suggestion": recommendation["suggestion"],
        "reason": recommendation["reason"],
    }


def _forecast_to_day(f: WeatherForecast) -> WeatherDayOut:
    """Convert internal WeatherForecast to API schema."""
    rec = get_weather_recommendation(f.symbol)
    return WeatherDayOut(
        date=f.date.isoformat(),
        symbol=f.symbol,
        icon=f.icon,
        label=f.label,
        temp_min=f.temp_min,
        temp_max=f.temp_max,
        precipitation_mm=f.precipitation_mm,
        wind_speed_ms=f.wind_speed_ms,
        indoor_suitable=f.is_indoor_suitable,
        outdoor_suitable=f.is_outdoor_suitable,
        suggestion=rec["suggestion"],
        suggestion_reason=rec["reason"],
    )
