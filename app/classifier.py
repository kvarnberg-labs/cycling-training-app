"""
Activity classification & pattern detection module.

Classifies cycling/training activities into workout types using
heuristics and detected patterns. Provides training pattern analysis
for LLM coaching context.

Classification levels:
- Workout type: endurance, threshold, vo2max, sweet_spot, tempo, recovery, race, commute, base, other
- Intensity zone: Z1-Z6 based on HR/power relative to FTP
- Pattern detection: back-to-back hard days, load spikes, recovery gaps

Usage:
    from app.classifier import ActivityClassifier
    classifier = ActivityClassifier(ftp=284, lthr=155)
    result = classifier.classify_all(activities)
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Workout type definitions ──

WORKOUT_TYPES = {
    "recovery": {
        "label": "Recovery",
        "power_range": (0, 55),
        "hr_range": (0, 65),
        "rpe_range": (1, 3),
        "keywords": ["recovery", "recover", "easy spin", "easy ride"],
        "description": "Very low intensity, active recovery session",
    },
    "endurance": {
        "label": "Endurance",
        "power_range": (55, 75),
        "hr_range": (65, 80),
        "rpe_range": (3, 5),
        "keywords": ["endurance", "road cycling", "long ride", "distance"],
        "description": "Zone 2 aerobic base building",
    },
    "tempo": {
        "label": "Tempo",
        "power_range": (76, 87),
        "hr_range": (78, 88),
        "rpe_range": (5, 6),
        "keywords": ["tempo", "steady", "moderate"],
        "description": "Steady moderate effort, solid aerobic work",
    },
    "sweet_spot": {
        "label": "Sweet Spot",
        "power_range": (88, 94),
        "hr_range": (85, 92),
        "rpe_range": (6, 7),
        "keywords": ["sweet spot", "sst", "ss"],
        "description": "Below threshold but high quality TSS-efficient work",
    },
    "threshold": {
        "label": "Threshold",
        "power_range": (95, 105),
        "hr_range": (90, 98),
        "rpe_range": (7, 8),
        "keywords": ["threshold", "ftp", "20min", "sweat", "suffer"],
        "description": "FTP/CP work, high sustainable intensity",
    },
    "vo2max": {
        "label": "VO2max",
        "power_range": (105, 120),
        "hr_range": (98, 102),
        "rpe_range": (8, 9),
        "keywords": ["vo2", "vo2max", "5min", "intervals", "hiit"],
        "description": "Short hard efforts to raise VO2 ceiling",
    },
    "anaerobic": {
        "label": "Anaerobic",
        "power_range": (120, 999),
        "hr_range": (100, 110),
        "rpe_range": (9, 10),
        "keywords": ["sprint", "anaerobic", "30s", "neuromuscular"],
        "description": "Short maximal efforts above VO2max",
    },
    "race": {
        "label": "Race",
        "power_range": (90, 150),
        "hr_range": (85, 110),
        "rpe_range": (8, 10),
        "keywords": ["race", "crit", "gran fondo", "sportive", "tävling"],
        "description": "Competitive event, unpredictable intensity",
    },
    "commute": {
        "label": "Commute",
        "power_range": (40, 65),
        "hr_range": (50, 70),
        "rpe_range": (2, 4),
        "keywords": ["commute", "pendling", "transport"],
        "description": "Transport riding, typically low intensity",
    },
    "base": {
        "label": "Base",
        "power_range": (55, 75),
        "hr_range": (65, 82),
        "rpe_range": (3, 5),
        "keywords": ["base", "foundation", "aerobic"],
        "description": "Structured endurance foundation work",
    },
    "undefined": {
        "label": "General",
        "power_range": (0, 999),
        "hr_range": (0, 999),
        "rpe_range": (1, 10),
        "keywords": [],
        "description": "Unclassified activity",
    },
}


class ActivityClassifier:
    """Classifies and analyses training activities.

    Args:
        ftp: Rider's functional threshold power in watts
        lthr: Rider's lactate threshold heart rate in bpm (optional)
        weight_kg: Rider's weight in kg (optional)
    """

    def __init__(
        self,
        ftp: Optional[float] = None,
        lthr: Optional[float] = None,
        weight_kg: Optional[float] = None,
    ):
        self.ftp = ftp or 250
        self.lthr = lthr
        self.weight_kg = weight_kg

    # ── Single activity classification ──

    def classify(self, activity: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a single activity into workout type and intensity zone.

        Args:
            activity: Activity dict from data_fetcher

        Returns:
            Activity dict with added 'classification' key
        """
        result = dict(activity)

        name = (activity.get("name") or "").lower()
        act_type = (activity.get("activity_type") or "").lower()
        hr = activity.get("average_heartrate")
        np = activity.get("weighted_avg_watts") or activity.get("average_watts")
        tss = activity.get("tss")
        rpe = activity.get("perceived_exertion")
        is_race = activity.get("race", False)
        is_commute = activity.get("commute", False)
        is_trainer = activity.get("trainer", False)
        duration_min = (activity.get("moving_time_seconds", 0) or 0) / 60

        # --- Determine workout type ---

        # Priority 1: Explicit flags
        if is_race:
            wtype = "race"
        elif is_commute:
            wtype = "commute"
        else:
            wtype = self._classify_by_name_and_data(
                name=name,
                act_type=act_type,
                hr=hr,
                np=np,
                tss=tss,
                duration_min=duration_min,
                rpe=rpe,
            )

        type_info = WORKOUT_TYPES.get(wtype, WORKOUT_TYPES["undefined"])

        # --- Determine intensity zone (power-based if available, else HR) ---
        if np and self.ftp:
            pct = (np / self.ftp) * 100
            zone, zone_label = self._power_zone(pct)
        elif hr and self.lthr:
            pct = (hr / self.lthr) * 100
            zone, zone_label = self._hr_zone(pct)
        elif hr and self.ftp:
            # Estimate: rough proxy using HR
            pct = (hr / max(180, (self.ftp * 0.65))) * 100
            zone, zone_label = self._hr_zone_abs(hr)
        else:
            zone = None
            zone_label = "unknown"

        # --- Intensity score (1-10) ---
        intensity = self._compute_intensity_score(wtype, zone, tss, duration_min)

        result["classification"] = {
            "workout_type": wtype,
            "workout_type_label": type_info["label"],
            "workout_type_description": type_info["description"],
            "intensity_zone": zone,
            "intensity_zone_label": zone_label,
            "intensity_score": intensity,
            "np_pct_ftp": round((np / self.ftp) * 100, 1) if np and self.ftp else None,
            "is_race": is_race,
            "is_commute": is_commute,
            "is_indoor": is_trainer,
            "classification_method": "power" if np and self.ftp else ("hr" if hr else "name"),
        }

        return result

    def _classify_by_name_and_data(
        self,
        name: str,
        act_type: str,
        hr: Optional[float],
        np: Optional[float],
        tss: Optional[float],
        duration_min: float,
        rpe: Optional[int],
    ) -> str:
        """Classify workout type using name heuristics + data signals."""
        # 1. Check name keywords (highest priority, specific first)
        # Priority order: most specific -> generic
        priority_types = ["race", "threshold", "vo2max", "anaerobic", "sweet_spot",
                          "recovery", "commute", "base", "tempo", "endurance"]
        for wtype in priority_types:
            info = WORKOUT_TYPES[wtype]
            for kw in info["keywords"]:
                if kw in name:
                    return wtype

        # 2. Check if it's a run (activity_type contains 'run')
        if "run" in act_type:
            # Check name for threshold/base indicators
            if "threshold" in name:
                return "threshold"
            elif "base" in name:
                return "base"
            elif "interval" in name or "vo2" in name:
                return "vo2max"
            else:
                return "endurance"

        # 3. Use NP vs FTP if available
        if np and self.ftp and np > 0:
            pct = np / self.ftp * 100
            if pct < 55:
                return "recovery"
            elif pct < 76:
                return "endurance"
            elif pct < 88:
                return "tempo"
            elif pct < 95:
                return "sweet_spot"
            elif pct < 106:
                return "threshold"
            elif pct < 125:
                return "vo2max"
            else:
                return "anaerobic"

        # 4. Use HR if available
        if hr:
            if hr < 120:
                return "endurance" if duration_min > 60 else "recovery"
            elif hr < 140:
                return "endurance"
            elif hr < 155:
                return "tempo"
            elif hr < 170:
                return "threshold"
            else:
                return "vo2max"

        # 5. Use duration + TSS for heuristic
        if duration_min < 45:
            return "threshold" if (tss or 0) > 50 else "recovery"
        elif duration_min > 120:
            return "endurance"
        elif duration_min > 60:
            return "tempo"
        else:
            return "endurance"

    @staticmethod
    def _power_zone(pct_ftp: float) -> Tuple[int, str]:
        """Classify power as percentage of FTP into zone."""
        if pct_ftp < 55:
            return (1, "Z1 - Active Recovery")
        elif pct_ftp < 76:
            return (2, "Z2 - Endurance")
        elif pct_ftp < 88:
            return (3, "Z3 - Tempo")
        elif pct_ftp < 95:
            return (4, "Sweet Spot")
        elif pct_ftp < 106:
            return (4, "Z4 - Threshold")
        elif pct_ftp < 125:
            return (5, "Z5 - VO2max")
        else:
            return (6, "Z6 - Anaerobic")

    @staticmethod
    def _hr_zone(pct_lthr: float) -> Tuple[int, str]:
        """Classify HR as percentage of LTHR into zone."""
        if pct_lthr < 80:
            return (1, "HR Z1")
        elif pct_lthr < 90:
            return (2, "HR Z2")
        elif pct_lthr < 95:
            return (3, "HR Z3")
        elif pct_lthr < 100:
            return (4, "HR Z4")
        else:
            return (5, "HR Z5")

    @staticmethod
    def _hr_zone_abs(hr: float) -> Tuple[int, str]:
        """Classify absolute HR into approximate zone."""
        if hr < 100:
            return (1, "HR Z1 - Very Easy")
        elif hr < 120:
            return (2, "HR Z2 - Easy")
        elif hr < 140:
            return (3, "HR Z3 - Moderate")
        elif hr < 160:
            return (4, "HR Z4 - Hard")
        else:
            return (5, "HR Z5 - Very Hard")

    @staticmethod
    def _compute_intensity_score(
        wtype: str, zone: Optional[int], tss: Optional[float], duration_min: float
    ) -> float:
        """Compute a 1-10 intensity score for the activity."""
        score = 5.0  # default moderate

        # Workout type base
        type_scores = {
            "recovery": 2, "commute": 2,
            "endurance": 4, "base": 4, "tempo": 5,
            "sweet_spot": 6, "threshold": 7,
            "vo2max": 8, "anaerobic": 9, "race": 8,
        }
        score = type_scores.get(wtype, 5)

        # Zone modifier
        if zone:
            if zone >= 5:
                score += 2
            elif zone >= 4:
                score += 1
            elif zone <= 2:
                score -= 1

        # TSS density bonus (high TSS in short time = hard)
        if tss and duration_min > 10:
            density = tss / (duration_min / 60)  # TSS per hour
            if density > 80:
                score += 1.5
            elif density > 60:
                score += 0.5
            elif density < 30:
                score -= 0.5

        return max(1, min(10, round(score, 1)))

    # ── Batch classification ──

    def classify_all(self, activities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify all activities and return with classification attached."""
        return [self.classify(a) for a in activities]

    # ── Pattern detection ──

    def detect_patterns(
        self,
        activities: List[Dict[str, Any]],
        days_back: int = 90,
    ) -> Dict[str, Any]:
        """Analyse activity list for training patterns and risk flags.

        Args:
            activities: List of classified activity dicts
            days_back: Lookback window for pattern detection

        Returns:
            Dict with pattern findings
        """
        # Sort by date
        sorted_acts = sorted(
            activities,
            key=lambda a: a.get("start_date", ""),
        )

        if len(sorted_acts) < 2:
            return {
                "total_activities": len(sorted_acts),
                "patterns": [],
                "risks": [],
                "summary": "Not enough data for pattern detection.",
            }

        patterns = []
        risks = []

        # --- Pattern 1: Back-to-back hard days ---
        consecutive_hard = self._find_consecutive_hard_days(sorted_acts)
        if consecutive_hard:
            patterns.append({
                "type": "consecutive_hard_days",
                "count": len(consecutive_hard),
                "events": consecutive_hard,
                "detail": "Back-to-back hard sessions detected. Ensure adequate recovery between hard efforts.",
            })
            if len(consecutive_hard) >= 2:
                risks.append({
                    "type": "overreaching",
                    "severity": "medium" if len(consecutive_hard) == 2 else "high",
                    "message": f"{len(consecutive_hard)} instances of consecutive hard days. Risk of accumulated fatigue.",
                })

        # --- Pattern 2: Weekly volume trend ---
        volume_trend = self._compute_volume_trend(sorted_acts)
        patterns.append(volume_trend)

        if volume_trend.get("weekly_change_pct", 0) > 15:
            risks.append({
                "type": "rapid_volume_increase",
                "severity": "medium",
                "message": f"Volume increased {volume_trend['weekly_change_pct']:.0f}% week-over-week. >10% is risky.",
            })
        elif volume_trend.get("weekly_change_pct", 0) < -30:
            risks.append({
                "type": "volume_drop",
                "severity": "low",
                "message": "Training volume dropped significantly. May indicate detraining or recovery.",
            })

        # --- Pattern 3: Recovery gap analysis ---
        recovery_gaps = self._find_recovery_gaps(sorted_acts)
        if recovery_gaps:
            patterns.append({
                "type": "recovery_gaps",
                "count": len(recovery_gaps),
                "events": recovery_gaps,
                "detail": "Periods with insufficient easy/recovery days between hard efforts.",
            })

        # --- Pattern 4: Commute vs training ratio ---
        commute_ratio = self._compute_commute_ratio(sorted_acts)
        if commute_ratio > 0.3:
            patterns.append({
                "type": "high_commute_ratio",
                "ratio": commute_ratio,
                "detail": f"{commute_ratio:.0%} of rides are commutes — factor into total load.",
            })

        # --- Pattern 5: Workout type distribution ---
        type_dist = self._compute_type_distribution(sorted_acts)
        patterns.append({
            "type": "workout_type_distribution",
            "distribution": type_dist,
            "detail": self._analyze_type_distribution(type_dist),
        })

        # --- Pattern 6: Rest days ---
        rest_days = self._find_rest_days(sorted_acts)
        patterns.append({
            "type": "rest_day_frequency",
            "rest_days": rest_days,
            "detail": f"{rest_days} rest days in period",
        })

        return {
            "total_activities": len(sorted_acts),
            "date_range": {
                "from": sorted_acts[0].get("start_date", "?")[:10] if sorted_acts[0].get("start_date") else "?",
                "to": sorted_acts[-1].get("start_date", "?")[:10] if sorted_acts[-1].get("start_date") else "?",
            },
            "patterns": patterns,
            "risks": risks,
            "workout_type_summary": type_dist,
            "summary": self._generate_summary(patterns, risks),
        }

    def _find_consecutive_hard_days(
        self, activities: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Find instances of hard efforts on consecutive days."""
        hard_types = {"threshold", "vo2max", "anaerobic", "race"}
        events = []

        for i in range(len(activities) - 1):
            curr = activities[i]
            next_act = activities[i + 1]

            c_class = curr.get("classification", {})
            n_class = next_act.get("classification", {})

            c_type = c_class.get("workout_type", "")
            n_type = n_class.get("workout_type", "")

            if c_type in hard_types and n_type in hard_types:
                c_date = curr.get("start_date", "?")[:10]
                n_date = next_act.get("start_date", "?")[:10]
                if c_date != n_date:
                    # Check if dates are consecutive
                    try:
                        cd = date.fromisoformat(c_date)
                        nd = date.fromisoformat(n_date)
                        if (nd - cd).days == 1:
                            events.append({
                                "date_1": c_date,
                                "date_2": n_date,
                                "type_1": c_type,
                                "type_2": n_type,
                                "name_1": curr.get("name", "?"),
                                "name_2": next_act.get("name", "?"),
                            })
                    except (ValueError, TypeError):
                        pass

        return events

    def _compute_volume_trend(
        self, activities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare recent week TSS to previous week TSS."""
        today = date.today()
        this_week = [a for a in activities if a.get("start_date", "") >= (today - timedelta(days=7)).isoformat()]
        last_week = [
            a for a in activities
            if (today - timedelta(days=14)).isoformat() <= a.get("start_date", "") < (today - timedelta(days=7)).isoformat()
        ]

        this_tss = sum(a.get("tss", 0) or 0 for a in this_week)
        last_tss = sum(a.get("tss", 0) or 0 for a in last_week)

        delta = this_tss - last_tss
        pct_change = (delta / last_tss * 100) if last_tss > 0 else 0

        return {
            "this_week_tss": round(this_tss, 0),
            "last_week_tss": round(last_tss, 0),
            "change": round(delta, 0),
            "weekly_change_pct": round(pct_change, 1),
            "trend": "increasing" if pct_change > 5 else ("decreasing" if pct_change < -5 else "stable"),
        }

    def _find_recovery_gaps(
        self, activities: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Find periods with 3+ consecutive hard days or missing recovery."""
        hard_types = {"threshold", "vo2max", "anaerobic", "race"}
        easy_types = {"recovery", "commute"}
        events = []

        hard_streak = 0
        streak_start = None

        for act in activities:
            c_type = act.get("classification", {}).get("workout_type", "")
            act_date = act.get("start_date", "?")[:10]

            if c_type in hard_types:
                if hard_streak == 0:
                    streak_start = act_date
                hard_streak += 1
            elif c_type not in easy_types:
                # Endurance/tempo days break the streak somewhat
                if hard_streak >= 3:
                    events.append({
                        "start": streak_start or "?",
                        "end": act_date,
                        "consecutive_hard_days": hard_streak,
                    })
                hard_streak = 0
                streak_start = None
            else:
                # Recovery/commute - resets
                if hard_streak >= 3:
                    events.append({
                        "start": streak_start or "?",
                        "end": act_date,
                        "consecutive_hard_days": hard_streak,
                    })
                hard_streak = 0
                streak_start = None

        return events

    def _compute_commute_ratio(self, activities: List[Dict[str, Any]]) -> float:
        """Compute ratio of commute rides to total rides."""
        total = len(activities)
        if total == 0:
            return 0
        commutes = sum(1 for a in activities if a.get("classification", {}).get("workout_type") == "commute")
        return commutes / total

    def _compute_type_distribution(
        self, activities: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Count activities by workout type."""
        dist: Dict[str, int] = {}
        for act in activities:
            wtype = act.get("classification", {}).get("workout_type", "undefined")
            dist[wtype] = dist.get(wtype, 0) + 1
        return dist

    def _find_rest_days(self, activities: List[Dict[str, Any]]) -> int:
        """Count rest days by finding gaps > 36h between activities."""
        if len(activities) < 2:
            return 0
        rest = 0
        dates = sorted(set(
            a.get("start_date", "")[:10] for a in activities if a.get("start_date")
        ))
        for i in range(len(dates) - 1):
            try:
                d1 = date.fromisoformat(dates[i])
                d2 = date.fromisoformat(dates[i + 1])
                gap = (d2 - d1).days
                if gap > 1:
                    rest += gap - 1
            except (ValueError, TypeError):
                pass
        return rest

    @staticmethod
    def _analyze_type_distribution(dist: Dict[str, int]) -> str:
        """Generate a human-readable analysis of workout type distribution."""
        total = sum(dist.values()) or 1
        hard = sum(dist.get(t, 0) for t in ("threshold", "vo2max", "anaerobic", "race"))
        easy = sum(dist.get(t, 0) for t in ("recovery", "commute", "endurance", "base"))
        hard_pct = hard / total * 100
        easy_pct = easy / total * 100

        if hard_pct > 40:
            return f"High intensity bias: {hard_pct:.0f}% hard workouts. Ensure adequate recovery."
        elif easy_pct > 80:
            return f"Endurance/recovery dominant ({easy_pct:.0f}%). Consider adding intensity if building."
        else:
            return f"Balanced mix: {hard_pct:.0f}% hard, {easy_pct:.0f}% easy. Good distribution."

    def _generate_summary(
        self, patterns: List[Dict[str, Any]], risks: List[Dict[str, Any]]
    ) -> str:
        """Generate a one-paragraph summary of patterns and risks."""
        parts = []
        if risks:
            parts.append(f"{len(risks)} risk(s) identified:")
            for r in risks:
                parts.append(f"  - [{r['severity']}] {r['message']}")
        else:
            parts.append("No significant risks detected.")
        return "\n".join(parts)
