"""
Coaching prompt templates — structured prompts designed to feed training
data into an LLM for professional cycling coaching recommendations.

Each template is a callable that accepts a data dict (from TrainingDataFetcher)
and returns a complete system + user prompt pair ready for LLM consumption.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional


# ── System identity ──

COACH_SYSTEM_PROMPT = """You are a professional cycling coach with 15+ years of experience coaching amateur to elite-level cyclists. You specialise in:

- **Periodization**: Base → Build → Peak → Race → Recovery cycles
- **PMC-based load management**: CTL (fitness), ATL (fatigue), TSB (form)
- **Power/HR zone training**: Zone-2 endurance, Sweet Spot, Threshold, VO2max, Anaerobic
- **Training load progression**: 10% rule, 3:1 hard/easy weeks, TSS ramp rates
- **Recovery management**: Active recovery, full rest, nutrition timing cues

Your coaching philosophy: Consistent, sustainable progression over hero sessions.
Train the rider in front of you — adapt to their life constraints, not an idealised plan.

YOU MUST:
- Be specific with power targets (watts) and HR targets (bpm)
- Give structured workout prescriptions with WARM-UP / MAIN SET / COOL-DOWN
- Explain WHY a workout fits the rider's current form
- Flag risks (overreaching, stale legs, life stress)
- Adapt to available data — if power is missing, prescribe by HR/RPE

YOU MUST NOT:
- Recommend unsafe ramp rates (>10% week-over-week TSS increase)
- Prescribe hard intervals when TSB < -10
- Ignore commute activities — factor them into total load
"""


# ── Helper: format data for template consumption ──


def _format_activity_for_prompt(a: Dict[str, Any]) -> str:
    """Format a single activity as a readable line for prompt context."""
    start = a.get("start_date", "?")[:10] if a.get("start_date") else "?"
    name = a.get("name", "?")
    act_type = a.get("activity_type", "Ride")
    dist = a.get("distance_km", 0)
    time = a.get("moving_time_seconds", 0) // 60
    tss = a.get("tss") or "-"
    hr = a.get("average_heartrate") or "-"
    np = a.get("weighted_avg_watts") or "-"
    ap = a.get("average_watts") or "-"
    elev = a.get("elevation_gain", 0)
    cad = a.get("average_cadence") or "-"
    commute = a.get("commute", False)
    race = a.get("race", False)
    trainer = a.get("trainer", False)
    rpe = a.get("perceived_exertion") or "-"
    ftp_during = a.get("rolling_ftp") or "-"

    tags = []
    if commute: tags.append("🚗 commute")
    if race: tags.append("🏁 race")
    if trainer: tags.append("🏠 indoor")
    tag_str = f" [{', '.join(tags)}]" if tags else ""

    return (
        f"{start} | {name:<40s} | {act_type:<10s} | "
        f"{time:>3d}min | {dist:>5.1f}km | "
        f"NP:{str(np):>4s} AP:{str(ap):>4s} | "
        f"HR:{str(hr):>3s} | TSS:{str(tss):>3s} | "
        f"Elev:{elev:>4.0f}m | Cad:{str(cad):>3s} | "
        f"RPE:{str(rpe):>2s} | FTP:{str(ftp_during):>3s}{tag_str}"
    )


def _format_pmc(pmc: Dict[str, Any]) -> str:
    """Format a single PMC entry."""
    return (f"  {pmc['date']}: CTL {pmc['fitness_ctl']:>4.0f}  "
            f"ATL {pmc['fatigue_atl']:>4.0f}  TSB {pmc['form_tsb']:>+5.0f}  "
            f"TSS {pmc['total_tss']:>4.0f}  Dist {pmc['total_distance_km']:>5.1f}km")


def build_context_block(data: Dict[str, Any]) -> str:
    """Build a structured context block from fetched data for prompt injection."""
    athlete = data.get("athlete", {})
    profile = data.get("training_overview", {})
    weekly = data.get("weekly_summary", {})
    activities = data.get("activities", [])
    pmc = data.get("pmc", [])

    ftp = athlete.get("ftp") or "(no FTP data)"
    weight = athlete.get("weight_kg") or "(no weight data)"

    lines = ["## ATHLETE PROFILE", f"- Name: {athlete.get('name', '?')}"]
    lines.append(f"- FTP: {ftp}W | Weight: {weight}kg | W/kg: {round(ftp / weight, 2) if ftp and weight else '?'}")
    lines.append(f"- Resting HR: {athlete.get('resting_hr', '?')} bpm | Max HR: {athlete.get('max_hr', '?')} bpm | LTHR: {athlete.get('lthr', '?')}")
    lines.append(f"- Timezone: {athlete.get('time_zone', '?')}")
    if athlete.get("power_zones"):
        lines.append(f"- Power Zones: {athlete['power_zones']}")
    if athlete.get("hr_zones"):
        lines.append(f"- HR Zones: {athlete['hr_zones']}")
    lines.append("")

    lines.append(f"## LATEST PMC ({pmc[-1]['date'] if pmc else 'N/A'})")
    if pmc:
        latest = pmc[-1]
        lines.append(f"- CTL (fitness): {latest['fitness_ctl']:.0f}")
        lines.append(f"- ATL (fatigue): {latest['fatigue_atl']:.0f}")
        lines.append(f"- TSB (form): {latest['form_tsb']:+.0f}")
        lines.append("")
        lines.append("### PMC History (weekly)")
        for p in pmc:
            lines.append(_format_pmc(p))
    lines.append("")

    lines.append(f"## RECENT WEEK SUMMARY (last {profile.get('days_back', '?')} days)")
    lines.append(f"- Rides: {weekly.get('ride_count', 0)} | Runs: {weekly.get('run_count', 0)}")
    lines.append(f"- Total distance: {weekly.get('total_distance_km', 0):.1f} km")
    lines.append(f"- Total TSS: {weekly.get('total_tss', 0)}")
    lines.append(f"- Total time: {weekly.get('total_time_minutes', 0)} min")
    lines.append(f"- Total elevation: {weekly.get('total_elevation_gain', 0):.0f} m")
    lines.append("")

    lines.append(f"## RECENT ACTIVITIES ({len(activities)} total)")
    for a in activities:
        lines.append(_format_activity_for_prompt(a))
    lines.append("")

    lines.append(f"## TODAY")
    lines.append(f"- Date: {date.today().isoformat()}")
    lines.append(f"- Day of week: {date.today().strftime('%A')}")
    lines.append("")

    return "\n".join(lines)


# ── Prompt template builders ──


def daily_workout_prompt(data: Dict[str, Any]) -> Dict[str, str]:
    """Build prompt for a single-day workout prescription.

    Returns {'system': ..., 'user': ...} prompt pair.
    """
    context = build_context_block(data)

    user_prompt = f"""Using the training data below, prescribe TODAY'S workout.

{context}

## REQUEST

Given the rider's current form (CTL/ATL/TSB), recent training load, and the day of week,
prescribe a specific workout for **today**.

YOUR RESPONSE MUST INCLUDE:
1. **Workout type** — recovery, endurance, sweet spot, threshold, VO2, or rest
2. **Duration** — total session time in minutes
3. **Structure** — warm-up, main set, cool-down with specific power/HR/RPE targets
4. **Why it fits** — explain how this connects to their current form and recent training
5. **RPE targets** — rate 1-10 for each segment
6. **Plan B** — what to do if legs feel unexpectedly bad/good

FORMAT AS:
```
## 🚴 [Workout Name]
**Duration:** XX min | **Focus:** [type] | **Difficulty:** [RPE X/10]

**Warm-up:** ...
**Main Set:** ...
**Cool-down:** ...

**Why:**
**Plan B:**
```"""

    return {
        "system": COACH_SYSTEM_PROMPT,
        "user": user_prompt,
    }


def weekly_plan_prompt(data: Dict[str, Any]) -> Dict[str, str]:
    """Build prompt for a full weekly training plan.

    Returns {'system': ..., 'user': ...} prompt pair.
    """
    context = build_context_block(data)

    user_prompt = f"""Using the training data below, design a WEEKLY TRAINING PLAN.

{context}

## REQUEST

Design a training plan for the **upcoming week** (starting today) based on the rider's
current form, recent training history, and the day of the week. This should be a
structured plan with specific workouts for each day.

YOUR RESPONSE MUST INCLUDE:
1. **Week theme** — what is the focus this week? (recovery, build, overload, etc.)
2. **Daily breakdown** — each day with workout type, duration, specific targets
3. **Expected load** — estimated TSS per day and for the week
4. **Progress check** — what to monitor to know if the week is going right
5. **Adjustment cues** — signs to skip, shorten, or extend sessions

FORMAT AS:
```
## 📅 Week of [date] — [Theme]

| Day | Workout | Duration | TSS est. | Focus |
|-----|---------|----------|----------|-------|
| Mon | ...     | ...      | ...      | ...   |
| Tue | ...     | ...      | ...      | ...   |
...

**Week total:** ~XXX TSS

**Notes:**
**Adjustments:**
```"""

    return {
        "system": COACH_SYSTEM_PROMPT,
        "user": user_prompt,
    }


def form_assessment_prompt(data: Dict[str, Any]) -> Dict[str, str]:
    """Build prompt for a form/fatigue assessment.

    Returns {'system': ..., 'user': ...} prompt pair.
    """
    context = build_context_block(data)

    user_prompt = f"""Using the training data below, assess the rider's current FORM AND READINESS.

{context}

## REQUEST

Analyse the rider's current training state in depth.

YOUR RESPONSE MUST COVER:
1. **Form assessment** — interpret CTL/ATL/TSB trends and what they mean for readiness
2. **Fatigue analysis** — identify any fatigue patterns, back-to-back hard days, accumulated load
3. **Training load trend** — is volume increasing, stable, or dropping? At an appropriate rate?
4. **Recovery status** — how recovered is the rider? Based on recent easy/rest days
5. **Risk assessment** — overtraining signals, stale form, burnout risk
6. **Training recommendation** — next 3-7 days direction based on this assessment
7. **Metrics to watch** — what specific numbers to track for the next week"""

    return {
        "system": COACH_SYSTEM_PROMPT,
        "user": user_prompt,
    }


def periodization_prompt(data: Dict[str, Any], target_event: str, target_date: str) -> Dict[str, str]:
    """Build prompt for a periodized training plan towards a target event.

    Args:
        data: Fetched training data
        target_event: What the rider is training for (e.g., "100km sportive", "Gran Fondo")
        target_date: Target date in YYYY-MM-DD format

    Returns {'system': ..., 'user': ...} prompt pair.
    """
    context = build_context_block(data)

    user_prompt = f"""Using the training data below, design a PERIODIZED TRAINING PLAN.

{context}

## REQUEST

Design a periodized training plan for the rider targeting the following event:

**Event:** {target_event}
**Date:** {target_date}

Weeks until event: approximately {(date.fromisoformat(target_date) - date.today()).days // 7 if target_date else "?"} weeks

YOUR RESPONSE MUST INCLUDE:
1. **Period breakdown** — phases: Base → Build → Peak → Taper → Race, with weekly TSS ranges
2. **Weekly structure** — example week for each phase (workout types, durations, TSS)
3. **Key workouts** — critical sessions that drive adaptation in each phase
4. **FTP progression** — realistic FTP improvement estimate across the plan
5. **Load progression** — weekly TSS ramp: starting point, peak, taper
6. **Recovery weeks** — when and how to deload
7. **Testing/reassessment** — when to retest FTP, power curve, etc."""

    return {
        "system": COACH_SYSTEM_PROMPT,
        "user": user_prompt,
    }


# ── Template registry ──

TEMPLATES = {
    "daily": daily_workout_prompt,
    "weekly": weekly_plan_prompt,
    "assessment": form_assessment_prompt,
    "periodization": periodization_prompt,
}


def get_template(name: str) -> callable:
    """Get a template builder by name."""
    if name not in TEMPLATES:
        raise KeyError(f"Unknown template '{name}'. Available: {list(TEMPLATES.keys())}")
    return TEMPLATES[name]
