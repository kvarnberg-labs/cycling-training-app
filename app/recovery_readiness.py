"""
Recovery Readiness Score — combines training load, form, and recovery
signals into a single 1-10 readiness score.

Factors:
  - TSB (form) — from PMC, weighted heavily
  - Recent training load — last 3 days TSS vs 7-day average
  - Rest days — how many in the last 4 days
  - Perceived exertion — from recent activities
  - Activity intensity — recent hard efforts

Output: 1-10 score with breakdown and recommendation.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RecoveryReadiness:
    """Calculate daily recovery readiness score.

    Args:
        tsb: Current TSB (form) value
        ctl: Current CTL (fitness) value
        atl: Current ATL (fatigue) value
        activities: List of activity dicts (recent 14 days)
    """

    def __init__(
        self,
        tsb: float = 0,
        ctl: float = 40,
        atl: float = 40,
        activities: Optional[List[Dict[str, Any]]] = None,
    ):
        self.tsb = tsb
        self.ctl = ctl
        self.atl = atl
        self.activities = activities or []

    def compute(self) -> Dict[str, Any]:
        """Compute the readiness score and return detailed breakdown.

        Returns dict with:
            score: 1-10 readiness score
            label: text label (e.g. "Ready", "Caution", "Rest")
            factors: breakdown of each factor's contribution
            recommendation: what to do today
        """
        today = date.today()
        four_days_ago = today - timedelta(days=4)
        seven_days_ago = today - timedelta(days=7)
        fourteen_days_ago = today - timedelta(days=14)

        # Filter recent activities
        recent_4 = [a for a in self.activities
                    if a.get("start_date", "") >= four_days_ago.isoformat()]
        recent_7 = [a for a in self.activities
                    if a.get("start_date", "") >= seven_days_ago.isoformat()]
        recent_14 = [a for a in self.activities
                     if a.get("start_date", "") >= fourteen_days_ago.isoformat()]

        # ── Factor 1: TSB (form) — 35% weight ──
        tsb_score, tsb_detail = self._score_tsb(self.tsb)
        tsb_weighted = tsb_score * 0.35

        # ── Factor 2: Recent training load — 25% weight ──
        tss_3d = sum(a.get("tss", 0) or 0 for a in recent_4)
        tss_7d = sum(a.get("tss", 0) or 0 for a in recent_7)
        load_score, load_detail = self._score_recent_load(tss_3d, tss_7d)
        load_weighted = load_score * 0.25

        # ── Factor 3: Rest days — 20% weight ──
        rest_days = 0
        # Count days in last 4 with no activity
        dates_with_activity = set()
        for a in recent_4:
            d = a.get("start_date", "")[:10]
            if d:
                dates_with_activity.add(d)
        for i in range(4):
            check = (today - timedelta(days=i)).isoformat()
            if check not in dates_with_activity:
                rest_days += 1

        rest_score, rest_detail = self._score_rest_days(rest_days)
        rest_weighted = rest_score * 0.20

        # ── Factor 4: Recent intensity — 20% weight ──
        hard_days_recent = sum(
            1 for a in recent_7
            if (a.get("classification") or {}).get("workout_type")
            in ("threshold", "vo2max", "anaerobic", "race")
            or (a.get("tss") or 0) > 80
        )
        intensity_score, intensity_detail = self._score_intensity(hard_days_recent)
        intensity_weighted = intensity_score * 0.20

        # ── Composite score ──
        raw_score = tsb_weighted + load_weighted + rest_weighted + intensity_weighted
        score = max(1, min(10, round(raw_score)))

        # ── Label ──
        label, recommendation = self._get_label_and_rec(score, tsb_score, rest_days)

        # ── Power adjustment ──
        power_adjustment_pct = self._recommend_power_adjustment(score)

        return {
            "score": score,
            "max_score": 10,
            "label": label,
            "recommendation": recommendation,
            "power_adjustment_pct": power_adjustment_pct,
            "factors": {
                "tsb_form": {
                    "score": round(tsb_score, 1),
                    "weight": 0.35,
                    "weighted_score": round(tsb_weighted, 1),
                    "detail": tsb_detail,
                    "value": round(self.tsb, 1),
                },
                "recent_load": {
                    "score": round(load_score, 1),
                    "weight": 0.25,
                    "weighted_score": round(load_weighted, 1),
                    "detail": load_detail,
                    "tss_3d": tss_3d,
                    "tss_7d": tss_7d,
                },
                "rest_days": {
                    "score": round(rest_score, 1),
                    "weight": 0.20,
                    "weighted_score": round(rest_weighted, 1),
                    "detail": rest_detail,
                    "days_without_activity": rest_days,
                },
                "recent_intensity": {
                    "score": round(intensity_weighted, 1),
                    "weight": 0.20,
                    "weighted_score": round(intensity_weighted, 1),
                    "detail": intensity_detail,
                    "hard_days": hard_days_recent,
                },
            },
            "pmc_snapshot": {
                "tsb": round(self.tsb, 1),
                "ctl": round(self.ctl, 1),
                "atl": round(self.atl, 1),
            },
        }

    @staticmethod
    def _score_tsb(tsb: float) -> Tuple[float, str]:
        """Score TSB on 1-10 scale. Higher TSB = higher readiness."""
        if tsb >= 15:
            return 10.0, "Very fresh — extended rest may indicate detraining"
        elif tsb >= 10:
            return 9.0, "Fresh — good form, room for intensity"
        elif tsb >= 5:
            return 8.0, "Slightly fresh — positive form, ready for work"
        elif tsb >= 0:
            return 7.0, "Neutral — balanced form, good for planned work"
        elif tsb >= -5:
            return 5.5, "Mildly fatigued — proceed with caution"
        elif tsb >= -10:
            return 4.0, "Fatigued — recovery day recommended"
        elif tsb >= -15:
            return 3.0, "Very fatigued — take it easy"
        elif tsb >= -20:
            return 2.0, "Deep fatigue hole — rest required"
        else:
            return 1.0, "Overtrained — extended rest essential"

    @staticmethod
    def _score_recent_load(tss_3d: float, tss_7d: float) -> Tuple[float, str]:
        """Score recent training load."""
        if tss_7d == 0:
            return 5.0, "No recent training data"
        recent_ratio = tss_3d / max(tss_7d, 1)  # ratio of last 3 days to 7 days

        if recent_ratio > 0.7:
            return 3.0, f"Heavy recent load ({tss_3d:.0f}TSS in 3d vs {tss_7d:.0f}TSS in 7d)"
        elif recent_ratio > 0.5:
            return 5.0, f"Moderate recent load ({tss_3d:.0f}TSS in 3d)"
        elif recent_ratio > 0.3:
            return 7.0, f"Light recent load ({tss_3d:.0f}TSS in 3d)"
        else:
            return 8.0, f"Very light recent load — well rested"

    @staticmethod
    def _score_rest_days(days: int) -> Tuple[float, str]:
        """Score based on rest days in last 4."""
        if days >= 3:
            return 10.0, "Excellent — 3+ rest days in last 4"
        elif days == 2:
            return 8.0, "Good — 2 rest days in last 4"
        elif days == 1:
            return 6.0, "Fair — 1 rest day in last 4"
        else:
            return 3.0, "No rest days in last 4"

    @staticmethod
    def _score_intensity(hard_days: int) -> Tuple[float, str]:
        """Score based on hard days in last 7."""
        if hard_days == 0:
            return 8.0, "No hard sessions recently — well rested"
        elif hard_days == 1:
            return 6.0, "1 hard session in last 7 — recovered"
        elif hard_days == 2:
            return 5.0, "2 hard sessions in last 7 — moderate load"
        elif hard_days == 3:
            return 4.0, "3 hard sessions in last 7 — accumulating fatigue"
        else:
            return 2.0, f"{hard_days} hard sessions in last 7 — high fatigue"

    @staticmethod
    def _get_label_and_rec(
        score: int, tsb_score: float, rest_days: int
    ) -> Tuple[str, str]:
        """Get label and training recommendation."""
        if score >= 9:
            return (
                "💪 Prime",
                "Excellent readiness. Ideal for threshold or VO2max work. "
                "Consider a high-intensity session or FTP test.",
            )
        elif score >= 7:
            return (
                "✅ Ready",
                "Good readiness. Planned training on track. "
                "Sweet Spot or endurance session appropriate.",
            )
        elif score >= 5:
            return (
                "⚠️ Caution",
                "Moderate readiness. Proceed with planned session "
                "but monitor legs during warm-up — be ready to dial intensity back.",
            )
        elif score >= 3:
            return (
                "🛑 Fatigued",
                "Low readiness. Recommend recovery ride or full rest day. "
                "If you must train, keep it Z1-Z2 and under 60 minutes.",
            )
        else:
            return (
                "🚨 Rest Required",
                "Very low readiness. Take a full rest day. "
                "Consider stretching, foam rolling, or light walking only.",
            )

    @staticmethod
    def _recommend_power_adjustment(score: int) -> int:
        """Recommend power target adjustment as percentage."""
        if score >= 8:
            return 100  # full power
        elif score >= 6:
            return 95  # 95% of normal targets
        elif score >= 4:
            return 85  # 85% of normal targets
        elif score >= 2:
            return 70  # 70% of normal targets
        else:
            return 50  # recovery only


def compute_readiness_from_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function — compute readiness from TrainingDataFetcher output.

    Args:
        data: Output from TrainingDataFetcher.fetch_all()

    Returns:
        Readiness score dict
    """
    pmc = data.get("pmc", [])
    activities = data.get("activities", [])

    tsb = 0
    ctl = 40
    atl = 40

    if pmc:
        latest = pmc[-1]
        tsb = latest.get("form_tsb", 0)
        ctl = latest.get("fitness_ctl", 40)
        atl = latest.get("fatigue_atl", 40)

    # Also check per-activity PMC data for more recent values
    sorted_acts = sorted(
        activities,
        key=lambda a: a.get("start_date", ""),
        reverse=True,
    )
    for a in sorted_acts[:3]:
        if a.get("fitness_ctl") and a.get("form_tsb") is not None:
            tsb = a.get("form_tsb", tsb)
            ctl = a.get("fitness_ctl", ctl)
            atl = a.get("fatigue_atl", atl)
            break

    readiness = RecoveryReadiness(
        tsb=tsb,
        ctl=ctl,
        atl=atl,
        activities=activities,
    )
    return readiness.compute()
