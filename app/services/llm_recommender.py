"""LLM-powered workout recommender using the OpenAI Agents SDK.

Takes Strava activity data, training metrics, and user profile as input,
constructs a rich context prompt, and uses an agent (via the OpenAI Agents SDK)
to generate structured weekly workout recommendations.

Configured to work with any OpenAI-compatible provider (OpenCode, OpenAI, etc.)
via LLM_API_KEY and LLM_API_BASE in .env.

Falls back to the rule-based engine if the agent is not configured.
"""

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from openai import AsyncOpenAI
from agents import Agent, Runner, set_default_openai_client, set_tracing_disabled

from app.config import settings

logger = logging.getLogger(__name__)


# ── Agent output models (structured output via the Agents SDK) ──


class WorkoutRecommendation(BaseModel):
    """A single workout recommendation for a specific day."""
    scheduled_date: str = Field(
        description="Date of the workout in YYYY-MM-DD format"
    )
    workout_type: str = Field(
        description="Type of workout: recovery, endurance, tempo, threshold, vo2max, sprint, or interval"
    )
    title: str = Field(
        description="Catchy, descriptive workout title"
    )
    description: str = Field(
        default="",
        description="Detailed workout description with specific intervals, durations, and targets"
    )
    duration_minutes: int = Field(
        default=60,
        description="Duration of the workout in minutes",
        ge=20,
        le=300,
    )
    target_power_zone: str = Field(
        default="",
        description="Power target zone description (e.g. 'Zone 2 (56-75% FTP)')"
    )
    target_rpe: Optional[int] = Field(
        default=None,
        description="Rate of Perceived Exertion on a 1-10 scale",
        ge=1,
        le=10,
    )
    is_indoor: bool = Field(
        default=False,
        description="Whether this should be an indoor workout (Zwift) due to weather"
    )


class WeeklyWorkoutPlan(BaseModel):
    """A weekly training plan with 3-6 recommended workouts."""
    workouts: List[WorkoutRecommendation] = Field(
        description="List of 3-6 workout recommendations for the week",
        min_length=1,
        max_length=10,
    )


# ── Agent definition ──

SYSTEM_PROMPT = """You are an expert cycling coach and sports scientist, similar to TrainingPeaks but smarter. Your specialty is designing personalized weekly training plans based on an athlete's actual Strava ride data, training load metrics, and goals.

You have deep knowledge of:
- Periodized training (base, build, peak, race, recovery phases)
- Power-based training zones and their physiological effects
- Training Stress Score (TSS), CTL/ATL/TSB (Performance Management Chart)
- Progressive overload, recovery management, and workout variety
- Weather-aware training decisions (indoor Zwift vs outdoor riding)
- Swedish/Nordic cycling conditions

Analyze the athlete's data below and design an optimal training week.

Available workout types and their purpose:
- recovery: Very easy spinning (Zone 1, RPE 2-3). 40-60 min. Active recovery, flush legs.
- endurance: Steady Zone 2 (56-75% FTP, RPE 3-4). 60-180 min. Aerobic base building.
- tempo: Sustained Zone 3 (76-87% FTP, RPE 5-6). 60-90 min. Muscular endurance.
- threshold: Near-FTP efforts (88-105% FTP, RPE 7-8). 50-80 min. Power at FTP.
- vo2max: Hard intervals (106-120% FTP, RPE 9). 45-65 min. Max aerobic power.
- sprint: All-out efforts (>120% FTP, RPE 10). 40-55 min. Neuromuscular power.
- interval: Mixed/variable intensity. 60-75 min. Race simulation, Fartlek.

Follow these principles when designing the plan:
1. Periodization: Match workout intensity/duration to the training phase
2. Progressive overload: Build load week-over-week where appropriate
3. Variety: Don't repeat the same workout type on consecutive days
4. Recovery: Schedule rest days or recovery rides between hard sessions
5. Weather awareness: Bad weather → suggest indoor/Zwift sessions
6. Realism: Workouts must be achievable given the athlete's current load and fatigue
7. Specificity: Write detailed, actionable descriptions with interval structures"""


# ── Context builder ──


def _build_athlete_context(
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
        Formatted context string for the agent input
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
        lines.append(f"Total rides synced: {len(recent_activities)}")
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

        # Weekly training summary
        weekly_stats = _summarize_weekly_training(recent_activities)
        lines.append("")
        lines.append("### Weekly Training Summary (Last 4 Weeks)")
        lines.append("| Week | Rides | Total TSS | Total Hours | Total Distance |")
        lines.append("|------|-------|-----------|-------------|----------------|")
        for ws in weekly_stats[-4:]:
            lines.append(
                f"| {ws['week_label']} | {ws['ride_count']} | "
                f"{ws['total_tss']:.0f} | {ws['total_hours']:.1f}h | "
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
            lines.append(
                f"| {day_str} | {label} | {temp} | {precip} | {wind} | {rec} |"
            )
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
    """Summarize activities by ISO week for training load overview."""
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


# ── Agent initialisation (lazy, with OpenCode provider) ──

_agent: Optional[Agent] = None
_client_initialised = False


def _ensure_agent() -> Optional[Agent]:
    """Initialise the OpenAI client and agent if configured.

    Uses the OpenCode provider endpoint (or any OpenAI-compatible API)
    configured via LLM_API_BASE and LLM_API_KEY in .env.

    Returns:
        The agent instance, or None if LLM is not configured.
    """
    global _agent, _client_initialised

    if _client_initialised:
        return _agent

    api_key = settings.llm_api_key
    api_base = settings.llm_api_base

    if not api_key or not api_base:
        logger.warning("LLM not configured — set LLM_API_KEY and LLM_API_BASE in .env")
        _client_initialised = True
        _agent = None
        return None

    # Configure the OpenAI client with the provider's endpoint
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=api_base,
    )

    # Set as the default client for the Agents SDK
    set_default_openai_client(client)
    # Disable tracing (no OpenAI trace API needed for custom endpoints)
    set_tracing_disabled(disabled=True)

    # Create the agent with structured output via Pydantic model
    _agent = Agent(
        name="Cycling Coach",
        instructions=SYSTEM_PROMPT,
        model=settings.llm_model,
        output_type=WeeklyWorkoutPlan,
    )

    _client_initialised = True
    logger.info(
        f"Agent SDK initialised: model={settings.llm_model}, "
        f"base_url={api_base}"
    )
    return _agent


# ── Public API ──


async def generate_llm_plan(
    user_id: int,
    user_profile: Dict[str, Any],
    training_metrics: Optional[Dict[str, float]],
    recent_activities: List[Dict[str, Any]],
    existing_scheduled: List[Dict[str, Any]],
    week_start: date,
    weather_forecasts: Optional[Dict[str, Dict]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Generate a weekly training plan using the agent.

    Uses the OpenAI Agents SDK with the configured provider (OpenCode, OpenAI, etc.)
    to produce structured workout recommendations.

    Args:
        user_id: The user's ID
        user_profile: User profile dict
        training_metrics: Latest CTL/ATL/TSB
        recent_activities: Recent Strava activities
        existing_scheduled: Already scheduled workouts for this week
        week_start: Monday of target week
        weather_forecasts: Weather forecast per day

    Returns:
        List of workout dicts (same format as the rule-based engine),
        or None if the agent is not configured or fails.
    """
    # Initialise the agent (lazy, once)
    agent = _ensure_agent()
    if agent is None:
        return None

    # Build athlete context
    athlete_context = _build_athlete_context(
        user_profile=user_profile,
        training_metrics=training_metrics,
        recent_activities=recent_activities,
        existing_scheduled=existing_scheduled,
        weather_forecasts=weather_forecasts,
    )

    # Build the user input with the task description
    week_end = week_start + timedelta(days=6)
    user_input = (
        f"Plan a training week from {week_start} to {week_end} for this athlete.\n\n"
        f"{athlete_context}\n\n"
        f"Design 3-6 workouts for this week. Output them as a WeeklyWorkoutPlan."
    )

    logger.info(
        "Running agent for workout recommendations "
        f"(model={settings.llm_model}, "
        f"activities={len(recent_activities)}, "
        f"provider={'configured' if settings.llm_api_base else 'not configured'})"
    )

    try:
        # Run the agent — the Agents SDK handles structured output parsing
        result = await Runner.run(
            agent,
            user_input,
        )

        # The structured output is a WeeklyWorkoutPlan
        plan = result.final_output
        if not isinstance(plan, WeeklyWorkoutPlan):
            logger.warning(
                f"Agent returned unexpected output type: {type(plan).__name__}"
            )
            return None

        if not plan.workouts:
            logger.warning("Agent returned empty workout list")
            return None

        # Convert to the standard dict format expected by the endpoint
        validated = []
        for w in plan.workouts:
            validated.append({
                "user_id": user_id,
                "scheduled_date": w.scheduled_date,
                "workout_type": w.workout_type,
                "title": w.title,
                "description": w.description or "",
                "duration_minutes": w.duration_minutes or 60,
                "target_power_zone": w.target_power_zone or "",
                "target_rpe": w.target_rpe,
                "status": "suggested",
                "source": "recommendation",
                "is_indoor": w.is_indoor,
            })

        logger.info(f"Agent generated {len(validated)} workouts")
        return validated

    except Exception as e:
        logger.error(f"Agent run failed: {e}")
        return None
