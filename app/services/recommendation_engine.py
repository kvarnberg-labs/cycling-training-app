"""Workout recommendation engine.

Recommends workouts based on:
- Current training load (CTL/ATL/TSB)
- Training goal/phase (base, build, peak, race, recovery)
- Recent workout history and workout type distribution
- Time available (duration preference)

Uses periodization principles:
  - Base phase: mostly Zone 2 endurance, some tempo
  - Build phase: threshold and VO2max work, less volume
  - Peak phase: race-specific intensity, decreased volume
  - Race phase: maintenance with rest before events
  - Recovery phase: easy spins, active recovery
"""

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import random
import math

from app.config import settings
from app.models import WorkoutType, TrainingGoal
from app.schemas import WorkoutOut


# ── Workout templates ──

WORKOUT_LIBRARY: Dict[str, List[Dict]] = {
    "recovery": [
        {"title": "Active Recovery Spin", "description": "Very easy spin. Keep HR in Zone 1. Focus on spinning legs, no force.", "duration_minutes": 45, "target_power_zone": "Zone 1 (<55% FTP)", "target_rpe": 2},
        {"title": "Recovery Ride", "description": "Flat route, easy gears. Stay conversational pace. Good for flushing legs.", "duration_minutes": 60, "target_power_zone": "Zone 1 (<55% FTP)", "target_rpe": 2},
        {"title": "Leg Opener", "description": "Short loosener. Include a few 30s spin-ups at 110+ RPM to wake up the legs.", "duration_minutes": 40, "target_power_zone": "Zone 1–2", "target_rpe": 3},
    ],
    "endurance": [
        {"title": "Endurance Ride", "description": "Steady Zone 2 effort. Keep power consistent. Long ride focus on aerobic development.", "duration_minutes": 120, "target_power_zone": "Zone 2 (56-75% FTP)", "target_rpe": 3},
        {"title": "Long Endurance", "description": "Extended Zone 2. Practice nutrition and pacing. Keep HR below threshold.", "duration_minutes": 180, "target_power_zone": "Zone 2 (56-75% FTP)", "target_rpe": 4},
        {"title": "Sweet Spot Base", "description": "Mixed endurance with sweet spot intervals: 3x15min at 88-93% FTP with 5min recoveries.", "duration_minutes": 90, "target_power_zone": "Sweet Spot (88-93% FTP)", "target_rpe": 5},
        {"title": "Aerobic Capacity", "description": "4x8min at Tempo/Low Threshold (80-90% FTP). Focus on smooth power delivery.", "duration_minutes": 75, "target_power_zone": "Tempo (76-87% FTP)", "target_rpe": 5},
    ],
    "tempo": [
        {"title": "Tempo Ride", "description": "Sustained effort in Zone 3. Builds muscular endurance. Stay aero.", "duration_minutes": 90, "target_power_zone": "Tempo (76-87% FTP)", "target_rpe": 5},
        {"title": "Progressive Tempo", "description": "Start Zone 2, build to Zone 3 by midpoint. Last 20min at upper tempo.", "duration_minutes": 90, "target_power_zone": "Tempo (76-87% FTP)", "target_rpe": 6},
        {"title": "Cruise Intervals", "description": "2x20min at 85-90% FTP with 5min rest. Classic threshold builder.", "duration_minutes": 75, "target_power_zone": "Sweet Spot/Tempo", "target_rpe": 6},
    ],
    "threshold": [
        {"title": "Threshold Intervals", "description": "3x12min at 95-105% FTP with 5min easy spin recoveries.", "duration_minutes": 75, "target_power_zone": "Threshold (88-105% FTP)", "target_rpe": 7},
        {"title": "Sweet Spot Session", "description": "2x20min at 88-95% FTP. Focus on aero position. Builds sustainable power.", "duration_minutes": 80, "target_power_zone": "Sweet Spot (88-95% FTP)", "target_rpe": 6},
        {"title": "Over/Under Intervals", "description": "3x(3min at 105% + 3min at 90% FTP) sets. Teaches you to handle surges at threshold.", "duration_minutes": 70, "target_power_zone": "Threshold/Vo2", "target_rpe": 8},
        {"title": "4x8 Threshold", "description": "4x8min at 98-105% FTP with 3min recoveries. High-quality threshold work.", "duration_minutes": 65, "target_power_zone": "Threshold", "target_rpe": 8},
    ],
    "vo2max": [
        {"title": "VO2 Max Intervals", "description": "4x4min at 110-120% FTP with 3min recoveries. Max aerobic power.", "duration_minutes": 60, "target_power_zone": "VO2 Max (106-120% FTP)", "target_rpe": 9},
        {"title": "Short Climbs", "description": "5x3min hard climbs at 110-120% FTP. Recover on descent.", "duration_minutes": 60, "target_power_zone": "VO2 Max", "target_rpe": 9},
        {"title": "Micro Intervals", "description": "6x2min at 115-125% FTP with 2min recoveries. Short and punchy.", "duration_minutes": 55, "target_power_zone": "VO2 Max", "target_rpe": 9},
        {"title": "Lactate Tolerance", "description": "5x3min at 110-115% FTP. Focus on holding power as fatigue builds.", "duration_minutes": 55, "target_power_zone": "Upper VO2", "target_rpe": 9},
    ],
    "sprint": [
        {"title": "Sprint Intervals", "description": "6x15s max effort sprints with 3min recoveries. Full gas.", "duration_minutes": 50, "target_power_zone": "Sprint (>120% FTP)", "target_rpe": 10},
        {"title": "Neuromuscular Power", "description": "8x10s max accelerations from slow roll. Force-focused.", "duration_minutes": 50, "target_power_zone": "Neuromuscular", "target_rpe": 10},
    ],
    "interval": [
        {"title": "Mixed Intervals", "description": "Pyramid: 1-2-3-2-1 min efforts with equal rest. Varied intensity from tempo to VO2.", "duration_minutes": 70, "target_power_zone": "Variable", "target_rpe": 7},
        {"title": "Fartlek Ride", "description": "Unstructured speed play. Surge between landmarks. Mimics race dynamics.", "duration_minutes": 75, "target_power_zone": "Variable", "target_rpe": 6},
    ],
}

# Phase distribution: what % of weekly workouts should be each type
PHASE_WORKOUT_SPLIT: Dict[TrainingGoal, Dict[str, float]] = {
    TrainingGoal.BASE: {
        "recovery": 0.15,
        "endurance": 0.50,
        "tempo": 0.25,
        "threshold": 0.10,
        "vo2max": 0.0,
        "sprint": 0.0,
        "interval": 0.0,
    },
    TrainingGoal.BUILD: {
        "recovery": 0.15,
        "endurance": 0.25,
        "tempo": 0.20,
        "threshold": 0.25,
        "vo2max": 0.10,
        "sprint": 0.0,
        "interval": 0.05,
    },
    TrainingGoal.PEAK: {
        "recovery": 0.20,
        "endurance": 0.10,
        "tempo": 0.10,
        "threshold": 0.25,
        "vo2max": 0.20,
        "sprint": 0.05,
        "interval": 0.10,
    },
    TrainingGoal.RACE: {
        "recovery": 0.25,
        "endurance": 0.10,
        "tempo": 0.10,
        "threshold": 0.20,
        "vo2max": 0.15,
        "sprint": 0.10,
        "interval": 0.10,
    },
    TrainingGoal.RECOVERY: {
        "recovery": 0.50,
        "endurance": 0.40,
        "tempo": 0.10,
        "threshold": 0.0,
        "vo2max": 0.0,
        "sprint": 0.0,
        "interval": 0.0,
    },
}

# Days since last workout type -> prioritize types not done recently
# Lower priority = more likely to recommend
WORKOUT_TYPE_PRIORITY = {
    "recovery": 0,
    "endurance": 1,
    "tempo": 2,
    "threshold": 3,
    "vo2max": 4,
    "sprint": 5,
    "interval": 3,
}

# TSB (form) ranges and corresponding workout intensity adjustments
TSB_RANGES = [
    (-100, -19, "overreaching"),   # Very fatigued — only recovery
    (-19, -9, "heavy"),             # Fatigued — easy endurance or recovery
    (-9, 6, "optimal"),            # Good form — can train hard
    (6, 16, "fresh"),              # Fresh — great for intensity
    (16, float("inf"), "peaking"), # Very fresh — race ready, don't overdo it
]


def _get_tsb_zone(tsb: float) -> str:
    """Determine training zone based on TSB (form)."""
    for lo, hi, zone in TSB_RANGES:
        if lo <= tsb < hi:
            return zone
    return "optimal"


def _get_workout_type_for_zone(tsb_zone: str, phase: TrainingGoal, recent_types: Dict[str, int]) -> str:
    """Pick a workout type based on TSB zone, training phase, and recent workout history.

    Args:
        tsb_zone: Current form zone (overreaching, heavy, optimal, fresh, peaking)
        phase: Current training phase/goal
        recent_types: Dict of {workout_type: days_since_last_that_type}

    Returns:
        Selected workout type string
    """
    phase_split = PHASE_WORKOUT_SPLIT.get(phase, PHASE_WORKOUT_SPLIT[TrainingGoal.BASE])

    # Filter available types based on TSB zone
    if tsb_zone == "overreaching":
        # Only recovery
        return "recovery"
    elif tsb_zone == "heavy":
        # Recovery or easy endurance
        choices = ["recovery", "endurance"]
        weights = [0.6, 0.4]
    elif tsb_zone == "peaking":
        # Light work, maintenance only
        choices = ["recovery", "endurance", "tempo"]
        weights = [0.4, 0.4, 0.2]
    else:
        # Normal training — use phase split
        choices = list(phase_split.keys())
        weights = [phase_split[c] for c in choices]

    # Adjust weights based on recent history: prioritize types not done recently
    for c in choices:
        days_since = recent_types.get(c, 99)
        if days_since < 1:
            # Already done this type today — deprioritize
            idx = choices.index(c)
            weights[idx] *= 0.1
        elif days_since < 3 and c in ("vo2max", "sprint", "threshold"):
            # High intensity types need more recovery between sessions
            idx = choices.index(c)
            weights[idx] *= 0.5

    # Normalize weights
    total = sum(weights)
    if total == 0:
        return "endurance"
    weights = [w / total for w in weights]

    # Weighted random pick
    r = random.random()
    cumulative = 0
    for choice, weight in zip(choices, weights):
        cumulative += weight
        if r <= cumulative:
            return choice

    return choices[-1]


def _pick_workout_from_library(workout_type: str, avoid_ids: set) -> Optional[Dict]:
    """Pick a workout from the library for the given type, avoiding duplicates.

    Args:
        workout_type: Type of workout to pick
        avoid_ids: Set of workout titles to avoid (already scheduled)

    Returns:
        Workout dict or None if none available
    """
    workouts = WORKOUT_LIBRARY.get(workout_type, [])
    if not workouts:
        return None

    available = [w for w in workouts if w["title"] not in avoid_ids]
    if not available:
        available = workouts  # Allow duplicates if we've used all variations

    chosen = random.choice(available)
    return {
        "title": chosen["title"],
        "description": chosen["description"],
        "duration_minutes": chosen["duration_minutes"],
        "target_power_zone": chosen["target_power_zone"],
        "target_rpe": chosen["target_rpe"],
        "workout_type": workout_type,
    }


def compute_weekly_tss_capacity(
    ctl: float,
    goal: TrainingGoal,
) -> Tuple[float, float]:
    """Compute recommended weekly TSS and max daily TSS.

    Based on CTL and training phase:
    - Weekly TSS target ≈ CTL * 7 (maintenance)
    - Build/Peak phases: 1.1-1.3x multiplier
    - Recovery phase: 0.5-0.7x multiplier

    Args:
        ctl: Current Chronic Training Load
        goal: Training goal/phase

    Returns:
        Tuple of (weekly_tss_target, max_daily_tss)
    """
    multipliers = {
        TrainingGoal.BASE: 0.9,
        TrainingGoal.BUILD: 1.1,
        TrainingGoal.PEAK: 1.2,
        TrainingGoal.RACE: 1.0,
        TrainingGoal.RECOVERY: 0.5,
    }
    multiplier = multipliers.get(goal, 1.0)

    # Minimum weekly TSS even for low CTL
    weekly_target = max(ctl * 7 * multiplier, 150)
    max_daily = weekly_target / 3  # Don't put more than 1/3 weekly load in one day

    return round(weekly_target, 0), round(max_daily, 0)


def generate_weekly_plan(
    user_id: int,
    ctl: float,
    atl: float,
    tsb: float,
    goal: TrainingGoal,
    ftp: int,
    recent_workouts: List[WorkoutOut],
    existing_scheduled: List[WorkoutOut],
    week_start: date,
) -> List[Dict]:
    """Generate a week of workout recommendations.

    Args:
        user_id: User ID
        ctl: Current CTL
        atl: Current ATL
        tsb: Current TSB
        goal: Current training goal/phase
        ftp: User's FTP
        recent_workouts: Recently completed workouts (for variety analysis)
        existing_scheduled: Already scheduled workouts for this week
        week_start: Monday of the target week

    Returns:
        List of workout dicts to recommend
    """
    tsb_zone = _get_tsb_zone(tsb)
    weekly_tss_target, max_daily_tss = compute_weekly_tss_capacity(ctl, goal)

    # Analyze recent workout types
    recent_types: Dict[str, int] = {}
    today = date.today()
    for w in recent_workouts:
        w_type = w.workout_type if hasattr(w, 'workout_type') and not w.workout_type else 'endurance'
        if w.scheduled_date:
            days_ago = (today - w.scheduled_date).days
            if w_type not in recent_types or days_ago < recent_types[w_type]:
                recent_types[w_type] = days_ago

    # Types already scheduled for this week
    scheduled_types = set()
    scheduled_titles = set()
    existing_tss = 0.0
    for w in existing_scheduled:
        scheduled_types.add(w.workout_type)
        scheduled_titles.add(w.title)
        existing_tss += w.actual_tss or w.actual_tss or 0

    # Pick the best 3-5 days for training (don't fill every day)
    days_to_schedule = _pick_training_days(week_start, tsb_zone, goal)

    recommendations = []
    remaining_tss = weekly_tss_target - existing_tss

    for day in days_to_schedule:
        if remaining_tss <= 0 and tsb_zone not in ("overreaching",):
            break

        # Pick workout type based on TSB, phase, and recent history
        w_type = _get_workout_type_for_zone(tsb_zone, goal, recent_types)

        # Get a workout from the library
        workout = _pick_workout_from_library(w_type, scheduled_titles)
        if not workout:
            continue

        # Estimate TSS for this workout
        estimated_tss = _estimate_workout_tss(workout, ftp)

        # Check daily TSS cap
        if estimated_tss > max_daily_tss and tsb_zone not in ("heavy", "overreaching"):
            # Scale down — pick a shorter/less intense version
            if w_type in ("vo2max", "sprint"):
                # Switch to endurance or tempo
                w_type = "tempo"
                workout = _pick_workout_from_library(w_type, scheduled_titles)
                if workout:
                    estimated_tss = _estimate_workout_tss(workout, ftp)
            else:
                # Shorter version of the same workout
                workout["duration_minutes"] = int(workout["duration_minutes"] * 0.7)
                estimated_tss = _estimate_workout_tss(workout, ftp)

        # Don't add if we'd exceed remaining TSS (but always add if TSB is healthy)
        is_overreaching = tsb_zone == "overreaching"
        if not is_overreaching and estimated_tss > remaining_tss + 20:
            continue

        suggestions = [w for k, v in WORKOUT_LIBRARY.items() for w in v]

        recommendations.append({
            "user_id": user_id,
            "scheduled_date": day,
            "workout_type": workout["workout_type"],
            "title": workout["title"],
            "description": workout["description"],
            "duration_minutes": workout["duration_minutes"],
            "target_power_zone": workout["target_power_zone"],
            "target_rpe": workout.get("target_rpe"),
            "status": "suggested",
            "source": "recommendation",
        })

        # Track what we've scheduled
        scheduled_types.add(workout["workout_type"])
        scheduled_titles.add(workout["title"])
        remaining_tss -= estimated_tss
        recent_types[workout["workout_type"]] = 0  # Just did this type

    return recommendations


def _estimate_workout_tss(workout: Dict, ftp: int) -> float:
    """Estimate TSS for a workout based on its type and duration.

    Rough TSS estimates per hour for different intensities:
      - Recovery: 30 TSS/hr
      - Endurance: 50 TSS/hr
      - Tempo: 75 TSS/hr
      - Threshold: 100 TSS/hr
      - VO2Max: 90 TSS/hr
      - Sprint: 60 TSS/hr

    Args:
        workout: Workout dict with type and duration
        ftp: User's FTP

    Returns:
        Estimated TSS
    """
    tss_per_hour = {
        "recovery": 30,
        "endurance": 50,
        "tempo": 75,
        "threshold": 100,
        "vo2max": 90,
        "sprint": 60,
        "interval": 70,
    }
    w_type = workout.get("workout_type", "endurance")
    rate = tss_per_hour.get(w_type, 50)
    hours = (workout.get("duration_minutes", 60) or 60) / 60.0
    return rate * hours


def _pick_training_days(week_start: date, tsb_zone: str, goal: TrainingGoal) -> List[date]:
    """Pick which days of the week to schedule training.

    Typical pattern for cyclists: train 4-5 days/week with rest days.
    Adjust based on TSB zone and training phase.

    Args:
        week_start: Monday of the target week
        tsb_zone: Current form zone
        goal: Training goal/phase

    Returns:
        List of dates to schedule workouts
    """
    base_days = {
        "overreaching": [3],  # Just one recovery ride mid-week
        "heavy": [1, 3, 5],   # Mon, Wed, Fri
        "optimal": [1, 2, 4, 6],  # Mon, Tue, Thu, Sat
        "fresh": [1, 2, 3, 5, 6],  # Mon, Tue, Wed, Fri, Sat
        "peaking": [1, 3, 5],  # Mon, Wed, Fri — lighter week
    }

    # Phase adjustments: add more days during build, fewer during race
    phase_adjustments = {
        TrainingGoal.BASE: 0,
        TrainingGoal.BUILD: 1,
        TrainingGoal.PEAK: 0,
        TrainingGoal.RACE: -1,
        TrainingGoal.RECOVERY: -1,
    }

    day_indices = base_days.get(tsb_zone, [1, 2, 4, 6])
    adjustment = phase_adjustments.get(goal, 0)

    if adjustment > 0:
        # Add Saturday (6) or Sunday (7) if not already in the list
        extra_days = [6, 7]
        for d in extra_days:
            if d not in day_indices and len(day_indices) < 6:
                day_indices.append(d)
    elif adjustment < 0:
        # Remove a day if we can
        if len(day_indices) > 3:
            day_indices.pop()

    day_indices.sort()
    return [week_start + timedelta(days=d - 1) for d in day_indices]
