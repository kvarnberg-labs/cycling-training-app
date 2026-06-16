"""
Coaching router — AI-agent-friendly endpoints for training data and
workout recommendations. Returns structured JSON designed for LLM
consumption, not browser rendering.

Endpoints:
    GET  /api/coaching/context          — Compact context pack for agent injection
    GET  /api/coaching/daily             — Daily workout prescription prompt
    GET  /api/coaching/weekly            — Weekly training plan prompt
    POST /api/coaching/periodization     — Periodization plan prompt
    GET  /api/coaching/activity-analysis — Analyse a specific activity
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/coaching", tags=["coaching"])


# ── Shared dependency: data fetcher ──


def _get_fetcher(user: User):
    """Get a data fetcher for the given user's credentials."""
    from app.data_fetcher import TrainingDataFetcher

    api_key = None
    athlete_id = None

    # Try per-user encrypted key first
    if user.intervals_api_key_encrypted:
        from app.services.encryption import decrypt
        decrypted = decrypt(user.intervals_api_key_encrypted)
        if decrypted:
            api_key = decrypted
            athlete_id = user.intervals_athlete_id

    # Fall back to server-level env config
    if not api_key:
        from app.config import settings
        api_key = settings.intervals_api_key
        athlete_id = settings.intervals_athlete_id

    if not api_key or not athlete_id:
        raise HTTPException(
            status_code=400,
            detail="Intervals.icu not configured. Add your API key in Settings.",
        )

    try:
        return TrainingDataFetcher(api_key=api_key, athlete_id=athlete_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Helper: wrap fetcher calls ──


def _fetch_or_404(fetcher_call):
    """Execute a fetcher call and wrap exceptions."""
    try:
        return fetcher_call()
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=502, detail=str(e))


# ── Endpoints ──


@router.get("/context")
def get_context(
    days_back: int = Query(42, description="Days of history"),
    compact: bool = Query(True, description="Compact mode (~800 chars)"),
    current_user: User = Depends(get_current_user),
):
    """Return compact context pack for AI agent injection.

    Designed to be fetched by an AI agent and injected into its
    system/context prompt. Returns markdown string with PMC snapshot,
    weekly summary, and recent activities.
    """
    fetcher = _get_fetcher(current_user)
    data = _fetch_or_404(lambda: fetcher.fetch_all(days_back=days_back))

    from app.context_pack import build_context_pack
    context = build_context_pack(data, compact=compact)

    return {
        "context": context,
        "context_chars": len(context),
        "fetched_at": data.get("fetched_at"),
        "athlete": data["athlete"].get("name"),
        "compact": compact,
    }


@router.get("/daily")
def get_daily_prompt(
    days_back: int = Query(42, description="Days of history for context"),
    current_user: User = Depends(get_current_user),
):
    """Return the daily workout prescription prompt.

    Returns the full prompt (system + user) that an LLM can consume
    directly to generate a personalised daily workout.
    """
    fetcher = _get_fetcher(current_user)
    data = _fetch_or_404(lambda: fetcher.fetch_all(days_back=days_back))

    from app.prompts.coaching_templates import daily_workout_prompt
    prompts = daily_workout_prompt(data)

    return {
        "template": "daily",
        "system_prompt": prompts["system"],
        "user_prompt": prompts["user"],
        "athlete": data["athlete"].get("name"),
        "fetched_at": data.get("fetched_at"),
    }


@router.get("/weekly")
def get_weekly_prompt(
    days_back: int = Query(42, description="Days of history for context"),
    current_user: User = Depends(get_current_user),
):
    """Return the weekly training plan prompt."""
    fetcher = _get_fetcher(current_user)
    data = _fetch_or_404(lambda: fetcher.fetch_all(days_back=days_back))

    from app.prompts.coaching_templates import weekly_plan_prompt
    prompts = weekly_plan_prompt(data)

    return {
        "template": "weekly",
        "system_prompt": prompts["system"],
        "user_prompt": prompts["user"],
        "athlete": data["athlete"].get("name"),
        "fetched_at": data.get("fetched_at"),
    }


@router.post("/periodization")
def get_periodization_prompt(
    body: Dict[str, Any],
    days_back: int = Query(42, description="Days of history for context"),
    current_user: User = Depends(get_current_user),
):
    """Return a periodization plan prompt for a target event.

    Request body:
    {
        "event": "Gran Fondo 100km",
        "target_date": "2026-08-15"
    }
    """
    target_event = body.get("event", "Target event")
    target_date = body.get("target_date", "2026-08-01")

    if not target_date or not _validate_date(target_date):
        raise HTTPException(status_code=400, detail="Invalid target_date. Use YYYY-MM-DD.")

    fetcher = _get_fetcher(current_user)
    data = _fetch_or_404(lambda: fetcher.fetch_all(days_back=days_back))

    from app.prompts.coaching_templates import periodization_prompt
    prompts = periodization_prompt(data, target_event=target_event, target_date=target_date)

    return {
        "template": "periodization",
        "target_event": target_event,
        "target_date": target_date,
        "system_prompt": prompts["system"],
        "user_prompt": prompts["user"],
        "athlete": data["athlete"].get("name"),
        "fetched_at": data.get("fetched_at"),
    }


@router.get("/activity-analysis")
def get_activity_analysis(
    activity_name: str = Query("", description="Filter by activity name (partial match)"),
    days_back: int = Query(90, description="Days of history to search"),
    current_user: User = Depends(get_current_user),
):
    """Fetch and return activities for analysis.

    Intended for agent consumption — returns structured activity data
    that an LLM can analyse for patterns, intensity distribution, etc.
    """
    fetcher = _get_fetcher(current_user)
    activities = _fetch_or_404(lambda: fetcher.get_activities(days_back=days_back))

    if activity_name:
        activities = [
            a for a in activities
            if activity_name.lower() in (a.get("name") or "").lower()
        ]

    return {
        "count": len(activities),
        "activities": activities,
        "fetched_at": datetime.now().isoformat(),
    }


# ── Helpers ──


def _validate_date(date_str: str) -> bool:
    try:
        date.fromisoformat(date_str)
        return True
    except (ValueError, TypeError):
        return False
