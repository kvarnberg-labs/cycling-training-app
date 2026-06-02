"""LLM-powered workout recommender.

Takes Strava activity data, training metrics, and user profile as input,
constructs a rich prompt with all that context, and calls an OpenAI-compatible
LLM API to generate structured weekly workout recommendations.

Falls back to the rule-based engine if the LLM is not configured.
"""

import json
import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings
from app.models import TrainingGoal
from app.schemas import StravaActivityOut, WorkoutOut

logger = logging.getLogger(__name__)


# ── Prompt template ──

SYSTEM_PROMPT = """You are an expert cycling coach and sports scientist, similar to TrainingPeaks but smarter. Your specialty is designing personalized weekly training plans based on an athlete's actual Strava ride data, training load metrics, and goals.

You have deep knowledge of:
- Periodized training (base, build, peak, race, recovery phases)
- Power-based training zones and their physiological effects
- Training Stress Score (TSS), CTL/ATL/TSB (Performance Management Chart)
- Progressive overload, recovery management, and workout variety
- Weather-aware training decisions (indoor Zwift vs outdoor riding)
- Swedish/Nordic cycling conditions

Analyze the athlete's data below and design an optimal training week."""

WORKOUT_LIBRARY_DESCRIPTION = """Available workout types and their purpose:
- recovery: Very easy spinning (Zone 1, RPE 2-3). 40-60 min. Active recovery, flush legs.
- endurance: Steady Zone 2 (56-75% FTP, RPE 3-4). 60-180 min. Aerobic base building.
- tempo: Sustained Zone 3 (76-87% FTP, RPE 5-6). 60-90 min. Muscular endurance.
- threshold: Near-FTP efforts (88-105% FTP, RPE 7-8). 50-80 min. Power at FTP.
- vo2max: Hard intervals (106-120% FTP, RPE 9). 45-65 min. Max aerobic power.
- sprint: All-out efforts (>120% FTP, RPE 10). 40-55 min. Neuromuscular power.
- interval: Mixed/variable intensity. 60-75 min. Race simulation, Fartlek."""


def build_athlete_context(
    user_profile: Dict[str, Any],
    training_metrics: Optional[Dict[str, float]],
    recent_activities: List[Dict[str, Any]],
    existing_scheduled: List[Dict[str, Any]],
    weather_forecasts: Optional[Dict[str, Dict]] = None,
) -> str:
    """Build a detailed context string about the athlete's current state.

    Args:
        user_profile: User profile dict (FTP, weight, training_goal, etc.)
        training_metrics: Latest CTL/ATL/TSB values
        recent_activities: List of recent Strava activities with key metrics
        existing_scheduled: Already scheduled workouts for the target week
        weather_forecasts: Optional dict of {date: {weather_info}}

    Returns:
        Formatted context string for the LLM prompt
    """
    lines = []

    # ── User Profile ──
    lines.append("## ATHLETE PROFILE")
    lines.append(f"- FTP: {user_profile.get('ftp', 200)} watts")
    lines.append(f"- Weight: {user_profile.get('weight_kg', 75)} kg")
    lines.append(
        f"- Training Goal/Phase: {user_profile.get('training_goal', 'base')}"
    )
    lines.append(f"- Resting HR: {user_profile.get('resting_hr', 60)} bpm")
    lines.append(f"- Max HR: {user_profile.get('max_hr', 185)} bpm")
    if user_profile.get("location_lat") and user_profile.get("location_lon"):
        lines.append(
            f"- Location: lat {user_profile['location_lat']}, "
            f"lon {user_profile['location_lon']}"
        )
    lines.append("")

    # ── Training Load Metrics ──
    lines.append("## CURRENT TRAINING LOAD (PMC)")
    if training_metrics:
        ctl = training_metrics.get("ctl", 0)
        atl = training_metrics.get("atl", 0)
        tsb = training_metrics.get("tsb", 0)
        lines.append(f"- CTL (Fitness): {ctl:.1f}")
        lines.append(f"- ATL (Fatigue): {atl:.1f}")
        lines.append(f"- TSB (Form): {tsb:.1f}")
        # Interpret TSB
        if tsb < -20:
            lines.append("- Status: Deep fatigue zone — prioritize recovery")
        elif tsb < -10:
            lines.append("- Status: Fatigued — train with caution")
        elif tsb < 5:
            lines.append("- Status: Optimal training range")
        elif tsb < 15:
            lines.append("- Status: Fresh — good for intensity")
        else:
            lines.append("- Status: Peaking — fresh, don't overdo it")
    else:
        lines.append("- No training metrics available (new athlete)")
    lines.append("")

    # ── Recent Strava Activities ──
    lines.append("## RECENT STRAVA ACTIVITIES (Last 30 Days)")
    if recent_activities:
        lines.append(
            f"Total rides synced: {len(recent_activities)}"
        )
        lines.append("")
        lines.append(
            "| Date | Type | Name | Duration | Distance | Avg Power | NP | "
            "Avg HR | Elevation | TSS | IF |"
        )
        lines.append(
            "|------|------|------|----------|----------|-----------|-----|"
            "-------|-----------|-----|----|"
        )
        for act in recent_activities:
            start = act.get("start_date", "")
            if isinstance(start, str) and len(start) > 10:
                start = start[:10]
            atype = act.get("activity_type", act.get("type", "Ride"))
            name = (act.get("name") or "Untitled")[:30]
            dur_m = (act.get("moving_time") or 0) / 60
            dist_km = (act.get("distance") or 0) / 1000
            avg_w = act.get("average_watts") or "-"
            np_w = act.get("weighted_average_watts") or "-"
            avg_hr = act.get("average_heartrate") or "-"
            elev = act.get("total_elevation_gain") or "-"
            tss = act.get("training_stress_score") or "-"
            if_val = act.get("intensity_factor") or "-"

            lines.append(
                f"| {start} | {atype} | {name} | "
                f"{dur_m:.0f}min | {dist_km:.1f}km | "
                f"{avg_w}w | {np_w}w | {avg_hr}bpm | "
                f"{elev}m | {tss} | {if_val} |"
            )

        # Compute weekly totals
        weekly_stats = _summarize_weekly_training(recent_activities)
        lines.append("")
        lines.append("### Weekly Training Summary (Last 4 Weeks)")
        lines.append(
            "| Week | Rides | Total TSS | Total Hours | Total Distance |"
        )
        lines.append(
            "|------|-------|-----------|-------------|----------------|"
        )
        for ws in weekly_stats[-4:]:
            lines.append(
                f"| {ws['week_label']} | {ws['ride_count']} | "
                f"{ws['total_tss']:.0f} | "
                f"{ws['total_hours']:.1f}h | "
                f"{ws['total_distance_km']:.0f}km |"
            )
    else:
        lines.append("- No recent Strava activities synced")
    lines.append("")

    # ── Weather Forecast ──
    if weather_forecasts:
        lines.append("## WEATHER FORECAST (Upcoming Week)")
        lines.append(
            "| Date | Conditions | Temp | Precip | Wind | Recommendation |"
        )
        lines.append(
            "|------|------------|------|--------|------|----------------|"
        )
        for day_str, w in sorted(weather_forecasts.items()):
            label = w.get("label", w.get("symbol", "?"))
            temp = f"{w.get('temp_min', '?')}..{w.get('temp_max', '?')}°C"
            precip = f"{w.get('precipitation_mm', '?')}mm"
            wind = f"{w.get('wind_speed_ms', '?')}m/s"
            rec = "Indoor 🏠" if w.get("indoor") else "Outdoor 🌳"
            lines.append(f"| {day_str} | {label} | {temp} | {precip} | {wind} | {rec} |")
        lines.append("")

    # ── Already Scheduled ──
    if existing_scheduled:
        lines.append("## ALREADY SCHEDULED (This Week)")
        lines.append("The athlete already has these workouts scheduled:")
        for w in existing_scheduled:
            d = w.get("scheduled_date", "?")
            t = w.get("title", "Untitled")
            wt = w.get("workout_type", "?")
            dur = w.get("duration_minutes", "?")
            lines.append(f"- {d}: {t} ({wt}, {dur}min)")
        lines.append("")

    return "\n".join(lines)


def _summarize_weekly_training(
    activities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Summarize activities by ISO week for training load overview.

    Args:
        activities: List of activity dicts

    Returns:
        List of dicts with weekly summary per ISO week
    """
    from collections import defaultdict
    import datetime

    weeks: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"ride_count": 0, "total_tss": 0.0, "total_hours": 0.0, "total_distance_km": 0.0}
    )

    for act in activities:
        start = act.get("start_date")
        if not start:
            continue
        if isinstance(start, str):
            try:
                dt = datetime.datetime.fromisoformat(start.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
        else:
            dt = start

        iso_year, iso_week, _ = dt.isocalendar()
        week_key = f"{iso_year}-W{iso_week:02d}"

        weeks[week_key]["ride_count"] += 1
        tss = act.get("training_stress_score") or 0
        weeks[week_key]["total_tss"] += tss
        dur_h = (act.get("moving_time") or 0) / 3600
        weeks[week_key]["total_hours"] += dur_h
        dist_km = (act.get("distance") or 0) / 1000
        weeks[week_key]["total_distance_km"] += dist_km

    result = []
    for week_key in sorted(weeks.keys()):
        result.append({
            "week_label": week_key,
            **weeks[week_key],
        })
    return result


def build_recommendation_prompt(
    week_start: date,
    athlete_context: str,
) -> List[Dict[str, str]]:
    """Build the full LLM prompt messages.

    Args:
        week_start: Monday of the target week
        athlete_context: Context string from build_athlete_context()

    Returns:
        List of message dicts for the chat API
    """
    week_end = week_start + timedelta(days=6)

    user_prompt = f"""Plan a training week from {week_start} to {week_end} for this athlete.

{athlete_context}

## TASK

Design 3-6 workouts for this week based on the athlete's training load, recent activity history, and goals.

Follow these principles:
1. **Periodization**: Match workout intensity/duration to the training phase (base/build/peak/race/recovery)
2. **Progressive overload**: Week-over-week progression in load where appropriate
3. **Variety**: Don't repeat the same workout type on consecutive days. Mix endurance, intensity, and recovery
4. **Recovery**: Schedule rest days or recovery rides between hard sessions
5. **Weather awareness**: Use the weather forecast to decide indoor vs outdoor. Bad weather → suggest Zwift/indoor sessions with adjusted descriptions
6. **Realism**: Workouts should be achievable given the athlete's current load and fatigue level
7. **Specificity**: Write detailed, actionable descriptions. Include specific interval structures where appropriate

{WORKOUT_LIBRARY_DESCRIPTION}

## OUTPUT FORMAT

Return ONLY a valid JSON array of workout objects (no markdown, no code fences):

```json
[
  {{
    "scheduled_date": "YYYY-MM-DD",
    "workout_type": "endurance|tempo|threshold|vo2max|sprint|recovery|interval",
    "title": "Catchy workout title",
    "description": "Detailed workout description with specific intervals, durations, and targets",
    "duration_minutes": 60,
    "target_power_zone": "Zone description or power target",
    "target_rpe": 3,
    "is_indoor": false
  }}
]
```

Requirements for the output:
- "scheduled_date" must be between {week_start} and {week_end}
- "workout_type" must be one of: recovery, endurance, tempo, threshold, vo2max, sprint, interval
- "target_rpe" must be an integer 1-10
- "target_power_zone" should describe the power target in cycling terms
- "is_indoor" should be true if weather data suggests indoor workout
- Include 3-6 workouts total
- Do NOT include any text outside the JSON array"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


async def call_llm(
    messages: List[Dict[str, str]],
) -> Optional[str]:
    """Call an OpenAI-compatible LLM API.

    Args:
        messages: List of chat messages

    Returns:
        Response text content, or None on failure
    """
    api_key = settings.llm_api_key
    api_base = settings.llm_api_base

    if not api_key or not api_base:
        logger.warning("LLM not configured — set LLM_API_KEY and LLM_API_BASE in .env")
        return None

    # Ensure base ends with /chat/completions
    url = api_base.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"

    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "max_tokens": settings.llm_max_tokens,
        "temperature": 0.7,
        "response_format": {"type": "json_object"},
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return content
    except httpx.HTTPStatusError as e:
        logger.error(f"LLM API HTTP error: {e.response.status_code} - {e.response.text[:200]}")
        return None
    except httpx.TimeoutException:
        logger.error("LLM API timeout after 60s")
        return None
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        return None


def parse_llm_response(response_text: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Parse the LLM response into a list of workout dicts.

    Tries multiple parsing strategies:
    1. Direct JSON parse of the response
    2. Extract JSON from markdown code blocks
    3. Try to find any JSON array in the text

    Args:
        response_text: Raw text from the LLM

    Returns:
        List of workout dicts, or None if parsing fails
    """
    import re

    if not response_text:
        return None

    strategies = [
        # Strategy 1: Direct JSON parse
        lambda t: json.loads(t),
        # Strategy 2: JSON inside code fences
        lambda t: json.loads(
            re.search(
                r"```(?:json)?\s*\n?(.*?)\n?```", t, re.DOTALL
            ).group(1)
        )
        if re.search(r"```(?:json)?\s*\n?(.*?)\n?```", t, re.DOTALL)
        else None,
        # Strategy 3: Find JSON array in text
        lambda t: json.loads(
            re.search(r"(\[[\s\S]*\])", t).group(1)
        )
        if re.search(r"(\[[\s\S]*\])", t)
        else None,
    ]

    for strategy in strategies:
        try:
            result = strategy(response_text)
            if isinstance(result, list):
                return result
            # Also handle if the response is wrapped in {"workouts": [...]}
            if isinstance(result, dict):
                for key in ("workouts", "recommendations", "plan", "schedule"):
                    if key in result and isinstance(result[key], list):
                        return result[key]
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue

    logger.error(f"Failed to parse LLM response: {response_text[:300]}")
    return None


def validate_workout(workout: Dict[str, Any]) -> bool:
    """Validate a parsed workout dict has all required fields.

    Args:
        workout: Parsed workout dict

    Returns:
        True if valid, False otherwise
    """
    required = ["scheduled_date", "workout_type", "title"]
    valid_types = {
        "recovery", "endurance", "tempo", "threshold",
        "vo2max", "sprint", "interval",
    }

    for field in required:
        if field not in workout or not workout[field]:
            logger.warning(f"Workout missing required field: {field}")
            return False

    wtype = workout.get("workout_type", "")
    if wtype not in valid_types:
        logger.warning(f"Invalid workout_type: {wtype}")
        return False

    return True


async def generate_llm_plan(
    user_id: int,
    user_profile: Dict[str, Any],
    training_metrics: Optional[Dict[str, float]],
    recent_activities: List[Dict[str, Any]],
    existing_scheduled: List[Dict[str, Any]],
    week_start: date,
    weather_forecasts: Optional[Dict[str, Dict]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Generate a weekly training plan using the LLM.

    This is the main entry point. Gathers context, calls the LLM,
    and returns structured workout recommendations.

    Args:
        user_id: The user's ID
        user_profile: User profile dict
        training_metrics: Latest CTL/ATL/TSB
        recent_activities: Recent Strava activities
        existing_scheduled: Already scheduled workouts for this week
        week_start: Monday of target week
        weather_forecasts: Weather forecast per day

    Returns:
        List of workout dicts, or None if LLM fails
    """
    # Build context
    athlete_context = build_athlete_context(
        user_profile=user_profile,
        training_metrics=training_metrics,
        recent_activities=recent_activities,
        existing_scheduled=existing_scheduled,
        weather_forecasts=weather_forecasts,
    )

    # Build prompt
    messages = build_recommendation_prompt(
        week_start=week_start,
        athlete_context=athlete_context,
    )

    logger.info(
        "Calling LLM for workout recommendations "
        f"(model={settings.llm_model}, "
        f"activities={len(recent_activities)}, "
        f"weeks_ahead=1)"
    )

    # Call LLM
    response_text = await call_llm(messages)
    if not response_text:
        logger.warning("LLM returned no response — falling back")
        return None

    # Parse response
    workouts = parse_llm_response(response_text)
    if not workouts:
        logger.warning("Failed to parse LLM response — falling back")
        return None

    # Validate and normalize
    validated = []
    for w in workouts:
        if not validate_workout(w):
            continue
        validated.append({
            "user_id": user_id,
            "scheduled_date": w["scheduled_date"],
            "workout_type": w["workout_type"],
            "title": w["title"],
            "description": w.get("description", ""),
            "duration_minutes": w.get("duration_minutes", 60),
            "target_power_zone": w.get("target_power_zone", ""),
            "target_rpe": w.get("target_rpe"),
            "status": "suggested",
            "source": "recommendation",
            "is_indoor": w.get("is_indoor", False),
        })

    if not validated:
        logger.warning("No valid workouts after parsing LLM response")
        return None

    logger.info(f"LLM generated {len(validated)} workouts")
    return validated
