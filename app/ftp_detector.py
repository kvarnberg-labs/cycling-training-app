"""
FTP Auto-Detection — analyses best efforts and power curve data to
estimate current FTP, track trends, and suggest updates.

Uses the standard relationship: FTP ≈ 95% of best 20-minute power,
or ≈ best 20-minute power from a maximal effort (for cyclists not
using a formal 20-min FTP test protocol).

Also supports detection from:
  - Ramp test results
  - Best 8-minute power × 0.90 (alternate)
  - Best 60-minute power (direct, for longer events)
  - Recent sweet spot / threshold intervals

Usage:
    from app.ftp_detector import FTPDetector
    detector = FTPDetector(activities=activities, power_curves=curve_data)
    result = detector.detect()
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Detection constants ──

FTP_PCT_20MIN = 0.95    # 95% of best 20-min power
FTP_PCT_8MIN = 0.90     # 90% of best 8-min power  
FTP_PCT_60MIN = 1.00    # 100% of best 60-min power (if available)
FTP_PCT_RAMP = 0.75     # 75% of ramp test peak power
MIN_CONFIDENCE_ACTIVITIES = 3  # minimum activities needed for confidence
MAX_ACTIVITY_AGE_DAYS = 90     # don't use activities older than this
BEST_EFFORT_DURATIONS = [5, 60, 300, 480, 1200, 3600]  # seconds: 5s, 1min, 5min, 8min, 20min, 60min


class FTPDetector:
    """Detect and track FTP from training data and power curves.

    Args:
        activities: List of activity dicts from data_fetcher
        power_curves: Power curve data from Intervals.icu (optional)
        current_ftp: Current FTP setting for comparison
    """

    def __init__(
        self,
        activities: Optional[List[Dict[str, Any]]] = None,
        power_curves: Optional[Dict[str, Any]] = None,
        current_ftp: Optional[float] = None,
    ):
        self.activities = activities or []
        self.power_curves = power_curves or {}
        self.current_ftp = current_ftp

    def detect(self) -> Dict[str, Any]:
        """Run all FTP detection methods and return the best estimate.

        Returns:
            Dict with estimated FTP, method used, confidence, and supporting data.
        """
        estimates = []

        # Method 1: Power curve 20-minute
        est_20min = self._from_power_curve(1200)
        if est_20min:
            estimates.append(("power_curve_20min", est_20min * FTP_PCT_20MIN, 0.8))

        # Method 2: Best 20-min effort from activities (looks at NP/AP for long rides)
        est_activity = self._from_best_activity_effort()
        if est_activity:
            estimates.append(("best_activity_effort", est_activity, 0.7))

        # Method 3: Ramp test detection
        est_ramp = self._from_ramp_test()
        if est_ramp:
            estimates.append(("ramp_test", est_ramp, 0.6))

        # Method 4: Best 8-minute (climbing)
        est_8min = self._from_power_curve(480)
        if est_8min:
            estimates.append(("power_curve_8min", est_8min * FTP_PCT_8MIN, 0.6))

        # Method 5: Rolling FTP from activities
        est_rolling = self._from_rolling_ftp()
        if est_rolling:
            estimates.append(("rolling_ftp", est_rolling, 0.5))

        if not estimates:
            return {
                "estimated_ftp": self.current_ftp,
                "ftp_source": "current_setting",
                "confidence": 0.3,
                "message": "Insufficient data for FTP detection. Using current FTP setting.",
                "methods_used": [],
                "all_estimates": [],
            }

        # Weighted average of all estimates
        total_weight = sum(w for _, _, w in estimates)
        weighted_ftp = sum(est * w for _, est, w in estimates) / total_weight

        # Confidence: based on how many methods agreed and data freshness
        confidence = self._compute_confidence(estimates)

        # Pick the best individual method
        best_method, best_ftp, best_weight = max(estimates, key=lambda x: x[1] * x[2])

        # Determine source
        if confidence >= 0.7:
            ftp_source = "auto_detected"
            message = f"FTP auto-detected at {weighted_ftp:.0f}W from {len(estimates)} data sources."
        elif confidence >= 0.4:
            ftp_source = "suggested"
            message = f"Suggested FTP: {weighted_ftp:.0f}W. Consider a formal FTP test to confirm."
        else:
            ftp_source = "low_confidence"
            message = f"Low-confidence estimate: {weighted_ftp:.0f}W. More data needed."

        # Generate trend
        trend = self._generate_trend(weighted_ftp)

        return {
            "estimated_ftp": round(weighted_ftp),
            "current_ftp": self.current_ftp,
            "delta": round(weighted_ftp - self.current_ftp) if self.current_ftp else None,
            "ftp_source": ftp_source,
            "confidence": round(confidence, 2),
            "message": message,
            "best_method": best_method,
            "best_method_ftp": round(best_ftp),
            "all_estimates": [
                {"method": m, "ftp": round(e), "weight": w}
                for m, e, w in estimates
            ],
            "trend": trend,
            "wkg": round(weighted_ftp / 78, 2) if weighted_ftp else None,
        }

    def _from_power_curve(self, duration_seconds: int) -> Optional[float]:
        """Extract best power for a duration from power curve data."""
        if not self.power_curves:
            return None

        # Intervals.icu format: {"list": [{"secs": [...], "values": [...], ...}, ...]}
        # Find the first (primary) power curve entry
        curve_list = self.power_curves.get("list", [])
        if not curve_list:
            return None

        primary = curve_list[0]
        secs = primary.get("secs", [])
        values = primary.get("values", [])

        if not secs or not values:
            return None

        # Find the closest duration
        best_val = None
        for s, v in zip(secs, values):
            if abs(s - duration_seconds) < 30 and v and v > 0:
                if best_val is None or abs(s - duration_seconds) < abs(best_val[0] - duration_seconds):
                    best_val = (s, v)

        if best_val:
            return float(best_val[1])

        # Also check powerModels for model-derived FTP
        power_models = primary.get("powerModels", [])
        for model in power_models:
            ftp = model.get("ftp")
            if ftp and ftp > 100:
                return float(ftp)

        return None

    def _from_best_activity_effort(self) -> Optional[float]:
        """Find best sustained power effort from recent activities."""
        if not self.activities:
            return None

        cutoff = (date.today() - timedelta(days=MAX_ACTIVITY_AGE_DAYS)).isoformat()
        recent = [
            a for a in self.activities
            if (a.get("start_date") or "")[:10] >= cutoff
        ]

        if len(recent) < MIN_CONFIDENCE_ACTIVITIES:
            return None

        candidates = []
        for a in recent:
            np = a.get("weighted_avg_watts") or a.get("average_watts")
            duration = a.get("moving_time_seconds", 0)
            tss = a.get("tss", 0)
            name = (a.get("name") or "").lower()

            if not np or duration < 600:  # need at least 10 min
                continue

            # High-quality candidates: threshold efforts, hard rides
            is_hard = (
                ("threshold" in name or "sweet spot" in name or "sst" in name)
                or (tss and tss > 60 and duration < 5400)  # high TSS density
                or a.get("classification", {}).get("workout_type")
                in ("threshold", "sweet_spot", "vo2max", "race")
            )

            weight = 1.0
            if is_hard:
                weight += 0.3
            if duration > 1200 and duration < 3600:  # 20-60 min — ideal
                weight += 0.2
            if tss and tss > 80:
                weight += 0.2

            candidates.append((np, weight))

        if not candidates:
            return None

        # Weighted average of top candidates
        sorted_cands = sorted(candidates, key=lambda x: x[0], reverse=True)[:5]
        total_w = sum(w for _, w in sorted_cands)
        return sum(np * w for np, w in sorted_cands) / total_w if total_w > 0 else None

    def _from_ramp_test(self) -> Optional[float]:
        """Detect FTP from ramp test patterns in activity names."""
        if not self.activities:
            return None

        for a in self.activities:
            name = (a.get("name") or "").lower()
            if "ramp" in name or "ftp test" in name or "20min test" in name:
                max_power = a.get("max_watts") or a.get("weighted_avg_watts")
                if max_power:
                    return max_power * FTP_PCT_RAMP
        return None

    def _from_rolling_ftp(self) -> Optional[float]:
        """Extract Intervals.icu's rolling FTP estimate from activities."""
        for a in self.activities:
            rolling = a.get("rolling_ftp")
            if rolling and rolling > 100:
                return float(rolling)
        return None

    def _compute_confidence(self, estimates: List[Tuple]) -> float:
        """Compute confidence score 0-1 based on agreement between methods."""
        if len(estimates) <= 1:
            return 0.4

        ftps = [e[1] for e in estimates]
        mean_ftp = sum(ftps) / len(ftps)

        # Coefficient of variation (lower = more agreement)
        variance = sum((f - mean_ftp) ** 2 for f in ftps) / len(ftps)
        cv = (variance ** 0.5) / mean_ftp if mean_ftp > 0 else 1

        # Low CV = high confidence
        if cv < 0.03:
            return 0.95
        elif cv < 0.05:
            return 0.85
        elif cv < 0.08:
            return 0.70
        elif cv < 0.12:
            return 0.55
        else:
            return 0.35

    def _generate_trend(self, estimated_ftp: float) -> Dict[str, Any]:
        """Compare estimate to current FTP and detect trend."""
        if not self.current_ftp:
            return {"direction": "unknown", "detail": "No baseline FTP for comparison."}

        delta = estimated_ftp - self.current_ftp
        abs_delta = abs(delta)
        pct = (delta / self.current_ftp) * 100

        if abs_delta < 3:
            direction = "stable"
            detail = f"FTP is stable at ~{self.current_ftp:.0f}W (within {abs_delta:.0f}W)."
        elif delta > 0:
            direction = "improving"
            detail = f"FTP may have improved by {abs_delta:.0f}W ({pct:.1f}%). Consider re-testing."
        else:
            direction = "declining"
            detail = f"FTP may have declined by {abs_delta:.0f}W ({pct:.1f}%). Could indicate accumulated fatigue or detraining."

        return {
            "direction": direction,
            "delta_watts": round(delta),
            "delta_pct": round(pct, 1),
            "detail": detail,
            "suggested_action": "Re-test FTP" if abs_delta > 5 else "Current FTP looks accurate",
        }
