"""Training load calculations — CTL, ATL, TSB (Banister model).

Based on the Performance Management Chart (PMC) model used by TrainingPeaks:
  - CTL (Chronic Training Load / Fitness): 42-day exponentially weighted moving average
  - ATL (Acute Training Load / Fatigue): 7-day exponentially weighted moving average
  - TSB (Training Stress Balance / Form): CTL - ATL

Formulas:
  CTL[t] = CTL[t-1] + (TSS[t] - CTL[t-1]) * (1 - exp(-1/42))
  ATL[t] = ATL[t-1] + (TSS[t] - ATL[t-1]) * (1 - exp(-1/7))
  TSB[t] = CTL[t] - ATL[t]

Constants: tau_ctl = 42, tau_atl = 7
"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
import math

TAU_CTL = 42  # Fitness time constant (days)
TAU_ATL = 7   # Fatigue time constant (days)


def exp_weighted_avg(previous: float, current_value: float, tau: float) -> float:
    """Calculate exponentially weighted moving average.

    Args:
        previous: Previous day's weighted average.
        current_value: Today's TSS value.
        tau: Time constant (in days).

    Returns:
        New weighted average.
    """
    alpha = 1 - math.exp(-1 / tau)
    return previous + (current_value - previous) * alpha


def calculate_tss(
    normalized_power: float,
    ftp: float,
    duration_seconds: float,
) -> float:
    """Calculate Training Stress Score.

    TSS = (duration_seconds * NP * IF) / (FTP * 3600) * 100

    where IF (Intensity Factor) = NP / FTP

    Simplified: TSS = (duration_seconds * NP * NP / FTP) / (FTP * 3600) * 100
    which reduces to: TSS = (sec * NP^2) / (FTP^2 * 36)

    If NP is not available, use average power.
    If no power data, estimate from heart rate or RPE.

    Args:
        normalized_power: Normalized Power (or average power) in watts
        ftp: Functional Threshold Power in watts
        duration_seconds: Activity duration in seconds

    Returns:
        Training Stress Score
    """
    if ftp <= 0 or duration_seconds <= 0 or normalized_power <= 0:
        return 0.0

    intensity_factor = normalized_power / ftp
    tss = (duration_seconds * normalized_power * intensity_factor) / (ftp * 3600) * 100
    return round(tss, 1)


def estimate_tss_from_hr(
    avg_hr: float,
    resting_hr: float,
    max_hr: float,
    duration_minutes: float,
    ftp: float,
    weight_kg: float,
) -> float:
    """Estimate TSS from heart rate data when power is unavailable.

    Uses a heart-rate-based approximation:
    - Calculate heart rate reserve (HRR) based % effort
    - Map to an estimated intensity factor
    - Derive TSS from estimated IF and duration

    This is a rough approximation — power-based TSS is always preferred.

    Args:
        avg_hr: Average heart rate during activity (bpm)
        resting_hr: Resting heart rate (bpm)
        max_hr: Maximum heart rate (bpm)
        duration_minutes: Activity duration in minutes
        ftp: Functional Threshold Power (watts)
        weight_kg: Rider weight (kg)

    Returns:
        Estimated TSS
    """
    if max_hr <= resting_hr or avg_hr <= resting_hr:
        return 0.0

    hr_reserve = max_hr - resting_hr
    hr_effort = (avg_hr - resting_hr) / hr_reserve

    # Rough mapping: HR effort to Intensity Factor
    # 60% HRR ≈ 0.70 IF (endurance), 80% HRR ≈ 0.90 IF (threshold), 90%+ ≈ 1.05+ IF
    estimated_if = 0.5 + hr_effort * 0.6
    estimated_if = min(estimated_if, 1.3)  # cap at reasonable max

    estimated_np = ftp * estimated_if
    estimated_tss = (
        (duration_minutes * 60) * estimated_np * estimated_if
    ) / (ftp * 3600) * 100

    return round(estimated_tss, 1)


def estimate_tss_from_rpe(
    rpe: int,
    duration_minutes: float,
    ftp: float,
    weight_kg: float,
) -> float:
    """Estimate TSS from Rate of Perceived Exertion (1-10, Borg CR-10).

    Session RPE (sRPE) method:
      Load = duration_minutes * RPE
      Rough mapping to TSS: TSS ≈ sRPE * 1.5-2.0

    Args:
        rpe: Rate of Perceived Exertion (1-10)
        duration_minutes: Duration in minutes
        ftp: Functional Threshold Power
        weight_kg: Rider weight

    Returns:
        Estimated TSS
    """
    if rpe <= 0 or duration_minutes <= 0:
        return 0.0

    # sRPE method with scaling
    srpe = duration_minutes * rpe
    estimated_tss = srpe * 1.5  # scaling factor
    return round(estimated_tss, 1)


def classify_workout_type(
    normalized_power: float,
    ftp: float,
    average_hr: Optional[float],
    max_hr: Optional[float],
    resting_hr: Optional[float],
) -> str:
    """Classify a ride into a workout type based on power and HR data.

    Power zones (Coggan):
      Zone 1: Active Recovery  — <55% FTP
      Zone 2: Endurance        — 56-75% FTP
      Zone 3: Tempo            — 76-87% FTP
      Zone 4: Threshold        — 88-105% FTP
      Zone 5: VO2 Max          — 106-120% FTP
      Zone 6: Anaerobic        — >120% FTP
      Zone 7: Neuromuscular    — sprints

    This classification looks at the average intensity to determine
    the primary workout type.

    Args:
        normalized_power: Normalized Power (watts)
        ftp: Functional Threshold Power (watts)
        average_hr: Average heart rate (optional)
        max_hr: Maximum heart rate (optional)
        resting_hr: Resting heart rate (optional)

    Returns:
        Workout type classification string
    """
    if ftp <= 0:
        return "endurance"

    intensity_factor = normalized_power / ftp

    if intensity_factor < 0.55:
        return "recovery"
    elif intensity_factor < 0.76:
        return "endurance"
    elif intensity_factor < 0.88:
        return "tempo"
    elif intensity_factor < 1.06:
        return "threshold"
    elif intensity_factor < 1.20:
        return "vo2max"
    else:
        return "sprint"


def pmc_series(
    daily_tss: Dict[date, float],
    num_days: int = 90,
) -> List[Tuple[date, float, float, float]]:
    """Calculate full PMC (Performance Management Chart) time series.

    Args:
        daily_tss: Dict mapping date -> total TSS for that day
        num_days: Number of days to compute

    Returns:
        List of (date, ctl, atl, tsb) tuples
    """
    if not daily_tss:
        return []

    dates = sorted(daily_tss.keys())
    if dates:
        start = dates[0]
    else:
        start = date.today() - timedelta(days=num_days)

    end = max(dates[-1], date.today())

    ctl = 0.0
    atl = 0.0
    series = []

    current = start
    while current <= end:
        tss = daily_tss.get(current, 0.0)
        ctl = exp_weighted_avg(ctl, tss, TAU_CTL)
        atl = exp_weighted_avg(atl, tss, TAU_ATL)
        tsb = ctl - atl
        series.append((current, round(ctl, 1), round(atl, 1), round(tsb, 1)))
        current += timedelta(days=1)

    return series
