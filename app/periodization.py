"""
Training periodization planner — builds phased training plans toward
a target event with progressive overload, recovery weeks, and taper.

Phases:
  BASE (6-12 weeks)    — Zone 2 volume, build aerobic engine
  BUILD (4-8 weeks)    — Add threshold/Sweet Spot, raise FTP
  PEAK (2-4 weeks)     — Race-specific intensity, VO2 work
  TAPER (1-2 weeks)    — Volume drop, intensity maintained
  RACE                  — Event
  RECOVERY (1-2 weeks)  — Active recovery post-event

Usage:
    from app.periodization import PeriodizationPlanner
    planner = PeriodizationPlanner(ftp=284, ctl=42, atl=34)
    plan = planner.build_plan(target_date="2026-09-15", weeks_available=...)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from enum import Enum
from math import ceil, floor
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Phase definitions ──


class Phase(Enum):
    BASE = "base"
    BUILD = "build"
    PEAK = "peak"
    TAPER = "taper"
    RACE = "race"
    RECOVERY = "recovery"


PHASE_CONFIG = {
    Phase.BASE: {
        "label": "Base",
        "min_weeks": 4,
        "max_weeks": 12,
        "description": "Aerobic foundation — build mitochondrial density, capillary network, and muscular endurance through sustained Zone 2 work.",
        "weekly_tss_pct_range": (0.6, 0.85),  # % of max weekly TSS
        "ftp_progression_pct": 2.0,  # % FTP gain per 4 weeks
        "intensity_split": {"endurance": 0.75, "tempo": 0.15, "sweet_spot": 0.10, "threshold": 0, "vo2max": 0},
        "hr_target_zones": [2, 3],
        "power_target_zones": [2, 3],
        "key_workout": "Long Zone 2 endurance ride (2-4h)",
        "recovery_frequency": "1-2 easy days per week, 1 full rest day",
    },
    Phase.BUILD: {
        "label": "Build",
        "min_weeks": 3,
        "max_weeks": 10,
        "description": "Raise FTP and lactate threshold through Sweet Spot and Threshold intervals. Maintain aerobic base.",
        "weekly_tss_pct_range": (0.75, 1.0),
        "ftp_progression_pct": 3.0,
        "intensity_split": {"endurance": 0.40, "tempo": 0.15, "sweet_spot": 0.25, "threshold": 0.15, "vo2max": 0.05},
        "hr_target_zones": [3, 4],
        "power_target_zones": [3, 4],
        "key_workout": "2×15min Sweet Spot @ 88-92% FTP",
        "recovery_frequency": "1 easy day per 2 hard days, 1 rest day per week",
    },
    Phase.PEAK: {
        "label": "Peak",
        "min_weeks": 2,
        "max_weeks": 6,
        "description": "Race specificity — VO2max work, over-under intervals, and race-pace efforts. Volume maintained or slightly reduced.",
        "weekly_tss_pct_range": (0.7, 0.95),
        "ftp_progression_pct": 1.0,
        "intensity_split": {"endurance": 0.35, "tempo": 0.10, "sweet_spot": 0.15, "threshold": 0.20, "vo2max": 0.20},
        "hr_target_zones": [4, 5],
        "power_target_zones": [4, 5],
        "key_workout": "3×5min VO2max @ 110-120% FTP",
        "recovery_frequency": "1 easy day between hard sessions, 1 rest day",
    },
    Phase.TAPER: {
        "label": "Taper",
        "min_weeks": 1,
        "max_weeks": 3,
        "description": "Reduce volume by 40-60% while maintaining intensity. Fresh legs for race day.",
        "weekly_tss_pct_range": (0.3, 0.5),
        "ftp_progression_pct": 0,
        "intensity_split": {"endurance": 0.40, "tempo": 0.10, "sweet_spot": 0.20, "threshold": 0.20, "vo2max": 0.10},
        "hr_target_zones": [2, 4],
        "power_target_zones": [2, 5],
        "key_workout": "Short race-pace efforts (3×3min @ target intensity)",
        "recovery_frequency": "Extra rest, short sessions",
    },
    Phase.RACE: {
        "label": "Race Week",
        "min_weeks": 1,
        "max_weeks": 1,
        "description": "Race week — minimal riding, peak freshness.",
        "weekly_tss_pct_range": (0.1, 0.3),
        "ftp_progression_pct": 0,
        "intensity_split": {"endurance": 0.30, "tempo": 0.10, "sweet_spot": 0.10, "threshold": 0.30, "vo2max": 0.20},
        "hr_target_zones": [1, 3],
        "power_target_zones": [1, 3],
        "key_workout": "Race day",
        "recovery_frequency": "Race + full rest",
    },
    Phase.RECOVERY: {
        "label": "Post-Event Recovery",
        "min_weeks": 1,
        "max_weeks": 3,
        "description": "Active recovery — easy spins, stretching, assess next goals.",
        "weekly_tss_pct_range": (0.3, 0.5),
        "ftp_progression_pct": 0,
        "intensity_split": {"endurance": 0.80, "tempo": 0.10, "recovery": 0.10, "sweet_spot": 0, "threshold": 0, "vo2max": 0},
        "hr_target_zones": [1, 2],
        "power_target_zones": [1, 2],
        "key_workout": "Easy spin 45-60 min, stretching",
        "recovery_frequency": "Daily easy if feeling good, skip if tired",
    },
}


@dataclass
class WeeklyPlan:
    """A single week in a periodized plan."""
    week_number: int
    phase: str
    week_label: str
    start_date: str
    end_date: str
    total_tss_target: int
    total_duration_minutes: int
    ride_count: int
    intensity_focus: str
    key_workout: str
    ftp_estimate: int
    is_recovery_week: bool
    notes: str = ""


@dataclass
class PeriodizedPlan:
    """Complete periodized training plan."""
    target_event: str
    target_date: str
    total_weeks: int
    start_date: str
    current_ftp: int
    current_ctl: float
    max_weekly_tss: int
    phases: List[Dict[str, Any]]
    weekly_plans: List[WeeklyPlan]
    ftp_projection: Dict[str, Any]
    volume_progression: List[Dict[str, Any]]


class PeriodizationPlanner:
    """Builds periodized training plans toward a target event.

    Args:
        ftp: Current FTP in watts
        ctl: Current CTL (chronic training load / fitness)
        atl: Current ATL (acute training load / fatigue)
        max_weekly_tss: Maximum sustainable weekly TSS (auto-calculated if None)
    """

    def __init__(
        self,
        ftp: float = 250,
        ctl: float = 40,
        atl: float = 40,
        max_weekly_tss: Optional[float] = None,
    ):
        self.ftp = ftp
        self.ctl = ctl
        self.atl = atl
        self.tsb = ctl - atl
        self._max_weekly_tss = max_weekly_tss

    @property
    def max_weekly_tss(self) -> float:
        """Maximum sustainable weekly TSS based on CTL."""
        if self._max_weekly_tss:
            return self._max_weekly_tss
        # Rule of thumb: max weekly TSS ≈ CTL × 10 with cap at ~700
        return min(max(self.ctl * 10, 300), 700)

    # ── Phase allocation ──

    def _allocate_phases(self, total_weeks: int) -> List[Tuple[Phase, int]]:
        """Distribute total weeks across training phases.

        Always normalises to exactly total_weeks — drops from build/peak first.
        """
        if total_weeks < 4:
            return [(Phase.BASE, max(1, total_weeks - 1)), (Phase.RACE, 1)]

        if total_weeks < 6:
            base = max(1, round(total_weeks * 0.35))
            build = max(1, round(total_weeks * 0.25))
            peak = max(1, total_weeks - base - build - 1)
            return [(Phase.BASE, base), (Phase.BUILD, build), (Phase.PEAK, peak), (Phase.TAPER, 1), (Phase.RACE, 1)]

        # Start with proportional allocation
        raw = {
            Phase.BASE: max(3, ceil(total_weeks * 0.35)),
            Phase.BUILD: max(2, ceil(total_weeks * 0.30)),
            Phase.PEAK: max(2, ceil(total_weeks * 0.20)),
            Phase.TAPER: max(1, ceil(total_weeks * 0.10)),
            Phase.RACE: 1,
        }

        # Normalise to exactly total_weeks
        allocated = sum(raw.values())
        diff = allocated - total_weeks
        if diff > 0:
            # Drop weeks from build, then peak, then base (last added, first cut)
            for phase in [Phase.PEAK, Phase.BUILD, Phase.BASE, Phase.TAPER]:
                cut = min(diff, raw[phase] - 1)
                raw[phase] -= cut
                diff -= cut
                if diff <= 0:
                    break
        elif diff < 0:
            # Add weeks to base (most flexible)
            raw[Phase.BASE] += abs(diff)

        return [(p, raw[p]) for p in [Phase.BASE, Phase.BUILD, Phase.PEAK, Phase.TAPER, Phase.RACE]]

    # ── Weekly TSS progression ──

    def _build_tss_progression(
        self, phase_alloc: List[Tuple[Phase, int]]
    ) -> List[Tuple[int, bool]]:
        """Build weekly TSS multipliers and recovery week flags.

        Returns list of (tss_multiplier_pct, is_recovery_week) for each week.
        Recovery weeks at 60% of surrounding load every 3-4 weeks.
        """
        total_weeks = sum(w for _, w in phase_alloc)
        profile = []
        week_in_phase = 0
        current_phase_tss_pct = 0.75  # start conservative

        for phase, weeks in phase_alloc:
            config = PHASE_CONFIG[phase]
            min_pct, max_pct = config["weekly_tss_pct_range"]

            for w in range(weeks):
                week_in_phase += 1
                is_recovery = False

                # Every 4th week is a recovery week (60% load)
                if week_in_phase % 4 == 0 and phase in (Phase.BASE, Phase.BUILD, Phase.PEAK):
                    is_recovery = True
                    mult = 0.6
                else:
                    # Ramp TSS within phase
                    phase_progress = w / max(weeks - 1, 1)
                    mult = min_pct + (max_pct - min_pct) * phase_progress

                profile.append((mult, is_recovery))

        return profile

    # ── Build the plan ──

    def build_plan(
        self,
        target_event: str = "Target event",
        target_date: Optional[str] = None,
        start_date: Optional[str] = None,
        conservative: bool = False,
    ) -> PeriodizedPlan:
        """Build a complete periodized training plan.

        Args:
            target_event: Name of the event
            target_date: Event date (YYYY-MM-DD). Defaults to 12 weeks from now.
            start_date: Plan start date (YYYY-MM-DD). Defaults to today.
            conservative: If True, use lower TSS ramp rates

        Returns:
            PeriodizedPlan dataclass
        """
        today = date.today()
        tgt = date.fromisoformat(target_date) if target_date else today + timedelta(weeks=12)
        start = date.fromisoformat(start_date) if start_date else today

        if tgt <= start:
            raise ValueError(f"Target date {tgt} must be after start date {start}")

        total_weeks = max(1, (tgt - start).days // 7)

        # Allocate phases
        phase_alloc = self._allocate_phases(total_weeks)
        tss_profile = self._build_tss_progression(phase_alloc)

        current_week_start = start
        weekly_plans = []
        phase_summaries = []
        ftp_estimates = [self.ftp]

        week_number = 1
        max_tss = self.max_weekly_tss * (0.85 if conservative else 1.0)

        for phase, num_weeks in phase_alloc:
            config = PHASE_CONFIG[phase]
            phase_week_start = week_number
            phase_plans = []

            phase_ftp_gain = config["ftp_progression_pct"]

            for w in range(num_weeks):
                global_idx = week_number - 1
                tss_mult, is_recovery = tss_profile[global_idx] if global_idx < len(tss_profile) else (0.75, False)

                weekly_tss = round(max_tss * tss_mult)
                weekly_duration = round(weekly_tss * 1.5)  # ~1.5 min per TSS point

                # Calculate FTP estimate
                ftp_progress = (week_number - 1) / max(total_weeks - 1, 1)
                ftp_gain_total = (phase_ftp_gain / 4) * (w + 1)  # per 4 weeks
                ftp_est = round(self.ftp * (1 + ftp_gain_total / 100))

                week_end = current_week_start + timedelta(days=6)
                plan = WeeklyPlan(
                    week_number=week_number,
                    phase=phase.value,
                    week_label=f"Week {week_number}",
                    start_date=current_week_start.isoformat(),
                    end_date=week_end.isoformat(),
                    total_tss_target=weekly_tss,
                    total_duration_minutes=weekly_duration,
                    ride_count=self._suggest_ride_count(weekly_tss, phase),
                    intensity_focus=self._get_intensity_label(phase, is_recovery),
                    key_workout=config["key_workout"],
                    ftp_estimate=ftp_est,
                    is_recovery_week=is_recovery,
                    notes="Recovery week — reduce intensity, focus on form" if is_recovery else "",
                )
                phase_plans.append(plan)
                weekly_plans.append(plan)
                current_week_start = week_end + timedelta(days=1)
                week_number += 1

            phase_summaries.append({
                "phase": phase.value,
                "label": config["label"],
                "description": config["description"],
                "start_week": phase_week_start,
                "end_week": week_number - 1,
                "num_weeks": num_weeks,
                "weekly_tss_range": f"{config['weekly_tss_pct_range'][0]*100:.0f}-{config['weekly_tss_pct_range'][1]*100:.0f}% of peak",
                "intensity_split": config["intensity_split"],
                "key_workout": config["key_workout"],
            })

        # FTP projection
        end_ftp = round(self.ftp * (1 + sum(PHASE_CONFIG[p]["ftp_progression_pct"] for p, _ in phase_alloc if p != Phase.RACE and p != Phase.TAPER and p != Phase.RECOVERY) / 400))

        ftp_projection = {
            "start_ftp": self.ftp,
            "projected_ftp": end_ftp,
            "gain_watts": end_ftp - self.ftp,
            "gain_pct": round((end_ftp - self.ftp) / self.ftp * 100, 1),
            "per_phase": {p.value: round(self.ftp * (1 + PHASE_CONFIG[p]["ftp_progression_pct"] * w / 400), 0)
                         for p, w in phase_alloc if p != Phase.RACE},
        }

        # Volume progression
        volume_progression = [
            {"week": p.week_number, "phase": p.phase, "tss": p.total_tss_target,
             "is_recovery": p.is_recovery_week}
            for p in weekly_plans
        ]

        return PeriodizedPlan(
            target_event=target_event,
            target_date=tgt.isoformat(),
            total_weeks=total_weeks,
            start_date=start.isoformat(),
            current_ftp=self.ftp,
            current_ctl=round(self.ctl, 1),
            max_weekly_tss=round(max_tss),
            phases=phase_summaries,
            weekly_plans=weekly_plans,
            ftp_projection=ftp_projection,
            volume_progression=volume_progression,
        )

    @staticmethod
    def _suggest_ride_count(weekly_tss: int, phase: Phase) -> int:
        """Suggest number of rides per week based on TSS target."""
        if phase == Phase.RACE:
            return 1
        if weekly_tss < 200:
            return 3
        elif weekly_tss < 350:
            return 4
        elif weekly_tss < 500:
            return 5
        else:
            return 5

    @staticmethod
    def _get_intensity_label(phase: Phase, is_recovery: bool) -> str:
        if is_recovery:
            return "Recovery — low intensity"
        labels = {
            Phase.BASE: "Aerobic endurance (Z2)",
            Phase.BUILD: "Sweet Spot / Threshold",
            Phase.PEAK: "VO2max / Race pace",
            Phase.TAPER: "Intensity maintained, volume reduced",
            Phase.RACE: "Minimal — peak freshness",
            Phase.RECOVERY: "Active recovery",
        }
        return labels.get(phase, "General")


# ── CLI helper ──


def format_plan_as_markdown(plan: PeriodizedPlan) -> str:
    """Format a PeriodizedPlan as a human-readable markdown table."""
    lines = [
        f"# 📅 {plan.target_event} — Training Plan",
        f"",
        f"**Start:** {plan.start_date} → **Event:** {plan.target_date}  |  **{plan.total_weeks} weeks total**",
        f"",
        f"## Rider Profile",
        f"- Current FTP: **{plan.current_ftp}W**",
        f"- Projected FTP: **{plan.ftp_projection['projected_ftp']}W** ({plan.ftp_projection['gain_pct']:+.1f}%, +{plan.ftp_projection['gain_watts']}W)",
        f"- Current CTL: {plan.current_ctl} | Max weekly TSS: {plan.max_weekly_tss}",
        f"",
        f"## Phase Breakdown",
        f"",
        f"| Phase | Weeks | Focus | Key Workout |",
        f"|-------|-------|-------|-------------|",
    ]
    for p in plan.phases:
        lines.append(
            f"| **{p['label']}** | {p['start_week']}-{p['end_week']} | "
            f"{p['description'][:80]}... | {p['key_workout']} |"
        )

    lines.extend([
        "",
        f"## Weekly Plan",
        f"",
        f"| Wk | Phase | TSS | Hours | Rides | Focus | Key Workout | FTP (est.) |",
        f"|---:|-------|----:|------:|------:|-------|-------------|----------:|",
    ])

    for w in plan.weekly_plans:
        rec = " 🔄" if w.is_recovery_week else ""
        lines.append(
            f"| {w.week_number:>2d} | {w.phase:<7s} | {w.total_tss_target:>3d} | "
            f"{w.total_duration_minutes//60:>2d}h{w.total_duration_minutes%60:02d} | {w.ride_count:>1d} | "
            f"{w.intensity_focus[:20]:<20s} | {w.key_workout[:30]:<30s} | {w.ftp_estimate}W |"
        )

    lines.append("")
    lines.append("---")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

    return "\n".join(lines)
