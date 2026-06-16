"""
Agent-ready context pack — compresses training data into a compact format
optimised for injection into an AI agent's system prompt.

The output is a ~1000-2000 character markdown block containing the most
salient training facts: PMC snapshot, power/HR zones, recent ride summary,
and training recommendations direction. Designed to keep context overhead
low while providing enough signal for intelligent decision-making.

Usage:
    from app.context_pack import build_context_pack
    context = build_context_pack(data)  # data from TrainingDataFetcher

CLI:
    python -m app.context_pack  # fetches latest data and prints context
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date, datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Helpers ──


def _fmt_tsb(tsb: float) -> str:
    """Format TSB with human-readable label."""
    if tsb < -15:
        return f"**{tsb:+.0f}** — 🚨 Deep fatigue, caution needed"
    elif tsb < -5:
        return f"**{tsb:+.0f}** — Building, manageable fatigue"
    elif tsb < 5:
        return f"**{tsb:+.0f}** — Neutral, good for intensity"
    elif tsb < 15:
        return f"**{tsb:+.0f}** — Fresh, possible detraining risk"
    else:
        return f"**{tsb:+.0f}** — Very fresh, extended rest may reduce fitness"


def _power_zone_label(watts: int, ftp: int) -> str:
    """Classify a power output into zone label."""
    if not ftp:
        return ""
    pct = watts / ftp * 100
    if pct < 55: return "Z1 (Active Recovery)"
    if pct < 75: return "Z2 (Endurance)"
    if pct < 88: return "Z3 (Tempo)"
    if pct < 94: return "Sweet Spot"
    if pct < 105: return "Z4 (Threshold)"
    if pct < 120: return "Z5 (VO2max)"
    return "Z6 (Anaerobic)"


def _weekday_label(d: str) -> str:
    """Convert ISO date to Swedish-friendly weekday."""
    if not d:
        return "?"
    try:
        dt = date.fromisoformat(d)
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        return days[dt.weekday()]
    except (ValueError, IndexError):
        return "?"


# ── Main builder ──


def build_context_pack(
    data: Dict[str, Any],
    compact: bool = True,
    max_recent_activities: int = 5,
) -> str:
    """Build a compact agent-injectable context pack from training data.

    Args:
        data: Output from TrainingDataFetcher.fetch_all()
        compact: If True, target ~1200 chars. If False, include more detail.
        max_recent_activities: Number of recent activities to include (0 = none)

    Returns:
        Markdown string suitable for LLM system prompt injection.
    """
    athlete = data.get("athlete", {})
    weekly = data.get("weekly_summary", {})
    activities = data.get("activities", [])
    pmc = data.get("pmc", [])

    ftp = athlete.get("ftp") or 0
    weight = athlete.get("weight_kg") or 0
    wpk = round(ftp / weight, 2) if ftp and weight else 0

    lines = []

    # ── Header ──
    lines.append("## 🚴 Training Context")
    lines.append("")

    # ── Athlete core ──
    lines.append(f"**Athlete:** {athlete.get('name', '?')} | "
                 f"**FTP:** {ftp}W | **W/kg:** {wpk}")

    if compact:
        lines.append(f"**Weight:** {weight}kg | **RHR:** {athlete.get('resting_hr', '?')}bpm")
    else:
        lines.append(f"**Weight:** {weight}kg | **RHR:** {athlete.get('resting_hr', '?')}bpm | "
                     f"**Max HR:** {athlete.get('max_hr', '?')}bpm | **LTHR:** {athlete.get('lthr', '?')}bpm")
        if athlete.get("power_zones"):
            lines.append(f"**Power Zones:** {athlete['power_zones']}")
        if athlete.get("hr_zones"):
            lines.append(f"**HR Zones:** {athlete['hr_zones']}")
    lines.append("")

    # ── PMC snapshot ──
    if pmc:
        latest = pmc[-1]
        prev = pmc[-2] if len(pmc) >= 2 else None

        ctl = latest["fitness_ctl"]
        atl = latest["fatigue_atl"]
        tsb = latest["form_tsb"]

        lines.append("**PMC (latest):**")
        lines.append(f"- CTL (Fitness): {ctl:.0f}")
        lines.append(f"- ATL (Fatigue): {atl:.0f}")
        lines.append(f"- TSB (Form): {_fmt_tsb(tsb)}")

        # Trend arrow
        if prev:
            ctl_delta = ctl - prev["fitness_ctl"]
            atl_delta = atl - prev["fatigue_atl"]
            ctl_arrow = "↑" if ctl_delta > 1 else ("↓" if ctl_delta < -1 else "→")
            atl_arrow = "↑" if atl_delta > 1 else ("↓" if atl_delta < -1 else "→")
            lines.append(f"- CTL trend: {ctl_arrow} ({ctl_delta:+.0f}/wk) | ATL trend: {atl_arrow} ({atl_delta:+.0f}/wk)")

        if not compact:
            lines.append(f"- Weekly data span: {pmc[0]['date']} → {pmc[-1]['date']}")
    lines.append("")

    # ── Recent week summary ──
    lines.append("**Recent Week:**")
    lines.append(f"- {weekly.get('ride_count', 0)} rides, {weekly.get('run_count', 0)} runs")
    lines.append(f"- {weekly.get('total_distance_km', 0):.0f} km | "
                 f"{weekly.get('total_time_minutes', 0)} min | "
                 f"{weekly.get('total_tss', 0)} TSS | "
                 f"{weekly.get('total_elevation_gain', 0):.0f}m ↑")
    lines.append("")

    # ── Recent activities (compact) ──
    if max_recent_activities > 0 and activities:
        # Sort by date descending, take N most recent
        sorted_acts = sorted(
            activities,
            key=lambda a: a.get("start_date", ""),
            reverse=True,
        )
        recent = sorted_acts[:max_recent_activities]

        lines.append("**Recent Activity Log:**")
        for a in recent:
            start = a.get("start_date", "?")[:10] if a.get("start_date") else "?"
            wd = _weekday_label(start)
            name = a.get("name", "?")[:25]
            act_type = a.get("activity_type", "Ride")
            dist = a.get("distance_km", 0)
            time = a.get("moving_time_seconds", 0) // 60
            tss = a.get("tss") or "-"
            hr = a.get("average_heartrate") or "-"
            np = a.get("weighted_avg_watts") or "-"
            tags = []
            if a.get("race"): tags.append("🏁")
            if a.get("commute"): tags.append("🚗")
            if a.get("is_strava_limited"): tags.append("⚠️limited")
            tag_str = " " + " ".join(tags) if tags else ""

            if compact:
                lines.append(f"- {wd} {start[5:]} | {name:25s} | {act_type:10s} | "
                             f"{time:>3d}min {dist:>4.0f}km | "
                             f"TSS:{str(tss):>3s} HR:{str(hr):>3s} NP:{str(np):>3s}{tag_str}")
            else:
                elev = a.get("elevation_gain", 0)
                lines.append(f"- {wd} {start} | {name:30s} | {act_type:10s} | "
                             f"{time:>4d}min | {dist:>5.1f}km | "
                             f"NP:{str(np):>3s} TSS:{str(tss):>3s} | "
                             f"HR:{str(hr):>3s} | {elev:>4.0f}m{tag_str}")
        lines.append("")

    # ── Training load assessment (non-compact only) ──
    if not compact and weekly.get("total_tss"):
        tss = weekly["total_tss"]
        if tss < 150:
            load_label = "Light"
        elif tss < 300:
            load_label = "Moderate"
        elif tss < 450:
            load_label = "Heavy"
        else:
            load_label = "Very heavy"
        lines.append(f"**Weekly load:** {tss} TSS ({load_label})")
        lines.append("")

    return "\n".join(lines)


# ── CLI ──


def main():
    """Fetch latest data and print the context pack."""
    try:
        from app.data_fetcher import TrainingDataFetcher
    except ImportError:
        print("ERROR: Must be run from the project root (cycling-training-app).", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("INTERVALS_API_KEY")
    athlete_id = os.environ.get("INTERVALS_ATHLETE_ID")

    if not api_key or not athlete_id:
        print("ERROR: Set INTERVALS_API_KEY and INTERVALS_ATHLETE_ID env vars.", file=sys.stderr)
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(description="Generate agent-ready training context pack")
    parser.add_argument("--compact", action="store_true", default=True, help="Compact ~1200 char output")
    parser.add_argument("--full", action="store_true", help="Detailed output (overrides --compact)")
    parser.add_argument("--days", type=int, default=42, help="Days of history")
    args = parser.parse_args()

    is_compact = not args.full

    try:
        fetcher = TrainingDataFetcher(api_key=api_key, athlete_id=athlete_id)
        data = fetcher.fetch_all(days_back=args.days)
        ctx = build_context_pack(data, compact=is_compact)
        print(ctx)
    except (ValueError, PermissionError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
