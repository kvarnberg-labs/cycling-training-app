"""
Workout Library & Templates — curated collection of structured workouts
organised by type, with power/HR/RPE targets as % of FTP.

Each template includes:
  - Name and type classification
  - Warm-up, main set, cool-down
  - Power targets (% of FTP)
  - HR targets (% of LTHR or absolute)
  - RPE targets
  - Duration estimates
  - Difficulty rating (1-10)
  - TSS estimate formula
  - Variants (easier/harder)

Usage:
    from app.workout_library import WorkoutLibrary, get_workout
    lib = WorkoutLibrary(ftp=284)
    workout = lib.get_workout("sweet_spot", duration_min=75)
    workouts = lib.find_workouts(type_filter="threshold", max_duration=60)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Workout duration definitions ──

WORKOUT_ZONES = {
    "z1": {"label": "Active Recovery", "power_pct": (0, 55), "hr_pct": (0, 65), "rpe": (1, 2)},
    "z2": {"label": "Endurance", "power_pct": (55, 75), "hr_pct": (65, 80), "rpe": (3, 4)},
    "z3": {"label": "Tempo", "power_pct": (76, 87), "hr_pct": (78, 88), "rpe": (5, 6)},
    "ss": {"label": "Sweet Spot", "power_pct": (88, 94), "hr_pct": (85, 92), "rpe": (6, 7)},
    "z4": {"label": "Threshold", "power_pct": (95, 105), "hr_pct": (90, 98), "rpe": (7, 8)},
    "z5": {"label": "VO2max", "power_pct": (105, 120), "hr_pct": (98, 102), "rpe": (8, 9)},
    "z6": {"label": "Anaerobic", "power_pct": (120, 150), "hr_pct": (100, 110), "rpe": (9, 10)},
    "neuromuscular": {"label": "Neuromuscular", "power_pct": (150, 999), "hr_pct": (0, 0), "rpe": (10, 10)},
}


@dataclass
class WorkoutSegment:
    """A single segment within a workout (warmup, interval, rest, cooldown)."""
    label: str
    duration_seconds: int
    power_pct_low: int
    power_pct_high: int
    cadence_target: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class WorkoutTemplate:
    """Complete workout template with structure and metadata."""
    id: str
    name: str
    workout_type: str  # recovery, endurance, tempo, sweet_spot, threshold, vo2max, anaerobic
    description: str
    duration_min: int
    difficulty: int  # 1-10
    estimated_tss: int
    segments: List[Dict[str, Any]] = field(default_factory=list)
    variants: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    indoor_compatible: bool = True
    outdoor_compatible: bool = True
    zwift_alternative: Optional[str] = None


# ── The library ──

RECOVERY_WORKOUTS = [
    WorkoutTemplate(
        id="rec-45", name="Recovery Spin", workout_type="recovery",
        description="Easy spin to flush legs and promote recovery. High cadence, very low intensity.",
        duration_min=45, difficulty=1, estimated_tss=25,
        segments=[
            {"label": "Easy Spin", "duration": 2700, "power_low": 0, "power_high": 55, "cadence": "90-100 rpm",
             "notes": "Stay in Z1. Keep HR under 110bpm if possible."},
            {"label": "Cool-down Stretch", "duration": 0, "power_low": 0, "power_high": 0,
             "notes": "Light stretching after ride", "off_bike": True},
        ],
        tags=["recovery", "easy", "flush"],
        zwift_alternative="Recovery Spin (pre-built)",
    ),
    WorkoutTemplate(
        id="rec-30", name="Micro Recovery", workout_type="recovery",
        description="Short recovery session for time-crunched days. Just enough to move the legs.",
        duration_min=30, difficulty=1, estimated_tss=15,
        segments=[
            {"label": "Easy Spin", "duration": 1800, "power_low": 0, "power_high": 55,
             "cadence": "90-100 rpm"},
        ],
        tags=["recovery", "short", "time-crunched"],
    ),
]

ENDURANCE_WORKOUTS = [
    WorkoutTemplate(
        id="end-120", name="Classic Endurance", workout_type="endurance",
        description="Aerobic base building. Steady Z2 effort, conversational pace.",
        duration_min=120, difficulty=3, estimated_tss=110,
        segments=[
            {"label": "Warm-up", "duration": 600, "power_low": 0, "power_high": 65, "cadence": "80-90 rpm",
             "notes": "Gradual build from easy to Z2"},
            {"label": "Main Set: Z2 Endurance", "duration": 6000, "power_low": 55, "power_high": 75,
             "cadence": "85-95 rpm",
             "notes": "Steady Z2 effort. Should be able to hold a conversation. Monitor HR between 65-80% LTHR."},
            {"label": "Cool-down", "duration": 600, "power_low": 0, "power_high": 55, "cadence": "80-90 rpm"},
        ],
        tags=["endurance", "aerobic", "base", "long"],
        zwift_alternative="Big Loop or Road to Sky",
    ),
    WorkoutTemplate(
        id="end-90", name="Midweek Endurance", workout_type="endurance",
        description="Solid midweek endurance session. Builds volume without excessive time commitment.",
        duration_min=90, difficulty=3, estimated_tss=80,
        segments=[
            {"label": "Warm-up", "duration": 600, "power_low": 0, "power_high": 65},
            {"label": "Main Set: Z2 Endurance", "duration": 4200, "power_low": 55, "power_high": 75,
             "cadence": "85-95 rpm"},
            {"label": "Tempo Finish (optional)", "duration": 300, "power_low": 76, "power_high": 87,
             "notes": "Last 5 min at tempo if feeling good"},
            {"label": "Cool-down", "duration": 300, "power_low": 0, "power_high": 55},
        ],
        tags=["endurance", "midweek"],
    ),
]

TEMPO_WORKOUTS = [
    WorkoutTemplate(
        id="tempo-60", name="Tempo Ride", workout_type="tempo",
        description="Solid tempo effort. Builds muscular endurance and improves lactate clearance.",
        duration_min=60, difficulty=5, estimated_tss=80,
        segments=[
            {"label": "Warm-up", "duration": 600, "power_low": 0, "power_high": 65},
            {"label": "Main Set: Tempo", "duration": 2400, "power_low": 76, "power_high": 87,
             "cadence": "85-95 rpm",
             "notes": "Steady tempo effort. 'Comfortably hard' — you can speak in short sentences."},
            {"label": "Easy Spinning", "duration": 300, "power_low": 0, "power_high": 55},
            {"label": "Main Set: Tempo", "duration": 600, "power_low": 76, "power_high": 87},
            {"label": "Cool-down", "duration": 300, "power_low": 0, "power_high": 55},
        ],
        tags=["tempo", "muscular endurance"],
        zwift_alternative="Temps Leger",
    ),
]

SWEET_SPOT_WORKOUTS = [
    WorkoutTemplate(
        id="ss-75", name="Sweet Spot Builder", workout_type="sweet_spot",
        description="High-quality TSS-efficient workout. Below threshold but highly productive.",
        duration_min=75, difficulty=6, estimated_tss=95,
        segments=[
            {"label": "Warm-up", "duration": 600, "power_low": 0, "power_high": 65,
             "notes": "Include 3×1min high cadence (100+ rpm) spin-ups"},
            {"label": "SS Interval 1", "duration": 720, "power_low": 88, "power_high": 94,
             "cadence": "85-95 rpm",
             "notes": "Sweet Spot effort. Breathing is deep but controlled."},
            {"label": "Rest", "duration": 180, "power_low": 0, "power_high": 55},
            {"label": "SS Interval 2", "duration": 720, "power_low": 88, "power_high": 94},
            {"label": "Rest", "duration": 180, "power_low": 0, "power_high": 55},
            {"label": "SS Interval 3", "duration": 720, "power_low": 88, "power_high": 94},
            {"label": "Cool-down", "duration": 300, "power_low": 0, "power_high": 55},
        ],
        tags=["sweet spot", "FTP builder", "quality"],
        zwift_alternative="The Gorby or McAdamic",
    ),
    WorkoutTemplate(
        id="ss-60", name="Sweet Spot Express", workout_type="sweet_spot",
        description="Shorter high-intensity Sweet Spot session for time-crunched days.",
        duration_min=60, difficulty=6, estimated_tss=75,
        segments=[
            {"label": "Warm-up", "duration": 600, "power_low": 0, "power_high": 65},
            {"label": "SS Interval 1", "duration": 600, "power_low": 88, "power_high": 94},
            {"label": "Rest", "duration": 180, "power_low": 0, "power_high": 55},
            {"label": "SS Interval 2", "duration": 600, "power_low": 88, "power_high": 94},
            {"label": "Cool-down", "duration": 300, "power_low": 0, "power_high": 55},
        ],
        tags=["sweet spot", "short"],
    ),
]

THRESHOLD_WORKOUTS = [
    WorkoutTemplate(
        id="thr-60", name="Threshold Intervals", workout_type="threshold",
        description="Classic threshold workout. Raises FTP and improves time-to-exhaustion at threshold.",
        duration_min=60, difficulty=7, estimated_tss=85,
        segments=[
            {"label": "Warm-up", "duration": 600, "power_low": 0, "power_high": 65,
             "notes": "Include 2×1min high cadence spin-ups"},
            {"label": "Threshold Interval 1", "duration": 480, "power_low": 95, "power_high": 105,
             "cadence": "85-95 rpm",
             "notes": "Hard but sustainable. You can say 1-2 words but not full sentences."},
            {"label": "Rest", "duration": 240, "power_low": 0, "power_high": 55},
            {"label": "Threshold Interval 2", "duration": 480, "power_low": 95, "power_high": 105},
            {"label": "Rest", "duration": 240, "power_low": 0, "power_high": 55},
            {"label": "Threshold Interval 3", "duration": 480, "power_low": 95, "power_high": 105},
            {"label": "Cool-down", "duration": 300, "power_low": 0, "power_high": 55},
        ],
        tags=["threshold", "FTP", "interval"],
        zwift_alternative="Mills or Dust",
    ),
    WorkoutTemplate(
        id="thr-40", name="Threshold Express", workout_type="threshold",
        description="Short sharp threshold session. High intensity density.",
        duration_min=40, difficulty=8, estimated_tss=65,
        segments=[
            {"label": "Warm-up", "duration": 600, "power_low": 0, "power_high": 65},
            {"label": "Threshold Intervals", "duration": 240, "power_low": 100, "power_high": 108,
             "cadence": "90-100 rpm", "notes": "3×4min @ 100-108% FTP, 3 min rest between"},
            {"label": "Rest", "duration": 180, "power_low": 0, "power_high": 55},
            {"label": "Threshold Intervals", "duration": 240, "power_low": 100, "power_high": 108},
            {"label": "Rest", "duration": 180, "power_low": 0, "power_high": 55},
            {"label": "Threshold Intervals", "duration": 240, "power_low": 100, "power_high": 108},
            {"label": "Cool-down", "duration": 120, "power_low": 0, "power_high": 55},
        ],
        tags=["threshold", "short", "intense"],
    ),
]

VO2MAX_WORKOUTS = [
    WorkoutTemplate(
        id="vo2-45", name="VO2max Intervals", workout_type="vo2max",
        description="Classic 3-minute VO2 intervals. Raises aerobic ceiling and improves high-end power.",
        duration_min=45, difficulty=8, estimated_tss=65,
        segments=[
            {"label": "Warm-up", "duration": 600, "power_low": 0, "power_high": 65,
             "notes": "15 min warm-up with 2×30s spin-ups at 110+ rpm"},
            {"label": "VO2 Interval 1", "duration": 180, "power_low": 105, "power_high": 120,
             "cadence": "90-100 rpm",
             "notes": "Hard but not maximal. Breathing is deep and rapid."},
            {"label": "Rest", "duration": 180, "power_low": 0, "power_high": 55},
            {"label": "VO2 Interval 2", "duration": 180, "power_low": 105, "power_high": 120},
            {"label": "Rest", "duration": 180, "power_low": 0, "power_high": 55},
            {"label": "VO2 Interval 3", "duration": 180, "power_low": 105, "power_high": 120},
            {"label": "Rest", "duration": 180, "power_low": 0, "power_high": 55},
            {"label": "VO2 Interval 4", "duration": 180, "power_low": 105, "power_high": 120},
            {"label": "Cool-down", "duration": 300, "power_low": 0, "power_high": 55},
        ],
        tags=["vo2max", "high intensity", "aerobic capacity"],
        zwift_alternative="Blue Ox (modified)",
    ),
    WorkoutTemplate(
        id="vo2-35", name="Short VO2 Bursts", workout_type="vo2max",
        description="Short, explosive VO2 efforts. Higher intensity but shorter intervals.",
        duration_min=35, difficulty=9, estimated_tss=55,
        segments=[
            {"label": "Warm-up", "duration": 600, "power_low": 0, "power_high": 65},
            {"label": "VO2 Interval 1", "duration": 120, "power_low": 110, "power_high": 130,
             "cadence": "95-105 rpm", "notes": "2 min hard. Give it everything you've got for the last 30s."},
            {"label": "Rest", "duration": 120, "power_low": 0, "power_high": 55},
            {"label": "VO2 Interval 2", "duration": 120, "power_low": 110, "power_high": 130},
            {"label": "Rest", "duration": 120, "power_low": 0, "power_high": 55},
            {"label": "VO2 Interval 3", "duration": 120, "power_low": 110, "power_high": 130},
            {"label": "Rest", "duration": 120, "power_low": 0, "power_high": 55},
            {"label": "VO2 Interval 4", "duration": 120, "power_low": 110, "power_high": 130},
            {"label": "Rest", "duration": 120, "power_low": 0, "power_high": 55},
            {"label": "VO2 Interval 5", "duration": 120, "power_low": 110, "power_high": 130},
            {"label": "Cool-down", "duration": 180, "power_low": 0, "power_high": 55},
        ],
        tags=["vo2max", "short", "explosive"],
    ),
]

ALL_WORKOUTS = (
    RECOVERY_WORKOUTS + ENDURANCE_WORKOUTS + TEMPO_WORKOUTS
    + SWEET_SPOT_WORKOUTS + THRESHOLD_WORKOUTS + VO2MAX_WORKOUTS
)

WORKOUT_MAP = {w.id: w for w in ALL_WORKOUTS}


class WorkoutLibrary:
    """Library of structured workout templates with FTP-based targets."""

    def __init__(self, ftp: float = 250, lthr: Optional[float] = None):
        self.ftp = ftp
        self.lthr = lthr

    def get_workout(self, workout_id: str) -> Optional[Dict[str, Any]]:
        """Get a workout by ID with FTP-specific power targets."""
        template = WORKOUT_MAP.get(workout_id)
        if not template:
            return None
        return self._render_workout(template)

    def get_workouts_by_type(
        self, workout_type: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get all workouts of a given type."""
        results = [w for w in ALL_WORKOUTS if w.workout_type == workout_type]
        return [self._render_workout(w) for w in results[:limit]]

    def find_workouts(
        self,
        type_filter: Optional[str] = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_difficulty: Optional[int] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find workouts matching criteria."""
        results = ALL_WORKOUTS

        if type_filter:
            results = [w for w in results if w.workout_type == type_filter]
        if max_duration:
            results = [w for w in results if w.duration_min <= max_duration]
        if min_duration:
            results = [w for w in results if w.duration_min >= min_duration]
        if max_difficulty:
            results = [w for w in results if w.difficulty <= max_difficulty]
        if tags:
            results = [w for w in results if any(t in w.tags for t in tags)]

        return [self._render_workout(w) for w in results[:limit]]

    def suggest_workout(
        self,
        readiness_score: int,
        duration_min: int = 60,
        preference: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Suggest the best workout based on readiness score.

        Args:
            readiness_score: 1-10 readiness score
            duration_min: Desired session length
            preference: Optional workout type preference

        Returns:
            Rendered workout dict
        """
        if readiness_score <= 2:
            type_filter = "recovery"
        elif readiness_score <= 4:
            type_filter = "recovery"
        elif readiness_score <= 6:
            type_filter = "endurance"
        elif readiness_score <= 8:
            type_filter = preference or "sweet_spot"
        else:
            type_filter = preference or "threshold"

        candidates = [
            w for w in ALL_WORKOUTS
            if w.workout_type == type_filter
            and abs(w.duration_min - duration_min) <= 30
        ]

        if not candidates:
            candidates = [w for w in ALL_WORKOUTS if w.workout_type == type_filter]

        if not candidates:
            candidates = [ALL_WORKOUTS[0]]

        return self._render_workout(candidates[0])

    def _render_workout(self, template: WorkoutTemplate) -> Dict[str, Any]:
        """Render a template with FTP-specific power targets."""
        ftp = self.ftp

        rendered_segments = []
        for seg in template.segments:
            if seg.get("off_bike"):
                rendered_segments.append(seg)
                continue

            power_low = round(seg["power_low"] / 100 * ftp) if seg["power_low"] > 0 else 0
            power_high = round(seg["power_high"] / 100 * ftp) if seg["power_high"] > 0 else 0

            dur_min = seg["duration"] // 60
            dur_sec = seg["duration"] % 60

            rendered_segments.append({
                "label": seg["label"],
                "duration": seg["duration"],
                "duration_str": f"{dur_min}:{dur_sec:02d}" if dur_min > 0 else f"{dur_sec}s",
                "power_range_watts": f"{power_low}-{power_high}W" if power_high > 0 else f"<{power_low}W",
                "power_pct_range": f"{seg['power_low']}-{seg['power_high']}%",
                "cadence": seg.get("cadence", "80-95 rpm"),
                "notes": seg.get("notes", ""),
                "rpe_range": self._estimate_rpe(seg["power_high"]),
            })

        return {
            "id": template.id,
            "name": template.name,
            "workout_type": template.workout_type,
            "description": template.description,
            "duration_min": template.duration_min,
            "difficulty": template.difficulty,
            "estimated_tss": template.estimated_tss,
            "segments": rendered_segments,
            "tags": template.tags,
            "ftp_setting": ftp,
            "zwift_alternative": template.zwift_alternative,
        }

    def _estimate_rpe(self, power_pct: int) -> str:
        """Estimate RPE from power percentage."""
        if power_pct <= 55: return "1-2"
        elif power_pct <= 75: return "3-4"
        elif power_pct <= 87: return "5-6"
        elif power_pct <= 94: return "6-7"
        elif power_pct <= 105: return "7-8"
        elif power_pct <= 120: return "8-9"
        else: return "9-10"
