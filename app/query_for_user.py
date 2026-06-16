"""
Per-User Training Query — get fitness advice for any registered Discord user.

The agent calls this tool when a Discord user asks for training advice.
It handles the full pipeline:
  1. Look up the user's Intervals.icu credentials (from Discord ID)
  2. Fetch their training data
  3. Build a coaching context with their training metrics
  4. Output a formatted recommendation (optionally with LLM)

Usage (for the Hermes agent):
    python -m app.query_for_user joey0624                        # Today's recommendation
    python -m app.query_for_user joey0624 --daily                 # Explicit daily
    python -m app.query_for_user joey0624 --weekly                # Weekly plan
    python -m app.query_for_user joey0624 --assessment            # Form assessment
    python -m app.query_for_user joey0624 --query "How was my week?"  # NL query
    python -m app.query_for_user --list-users                     # List registered users

All output goes to stdout so the agent can capture it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from app.user_manager import (
    get_training_data_for_user,
    list_discord_users,
    get_user_by_discord,
)

logger = logging.getLogger(__name__)


# ── Formatters ──


def _fmt_header(title: str) -> str:
    sep = "=" * 60
    return f"{sep}\n{title}\n{sep}"


def _fmt_section(title: str, body: str) -> str:
    return f"\n── {title} ──\n{body}"


def _format_recommendation(
    data: Dict[str, Any],
    template_type: str = "daily",
) -> str:
    """Build a human-readable fitness recommendation from training data.

    This is the LLM-free path — it builds a structured prompt/context
    that the Hermes agent can feed to its own model for the final answer.
    """
    athlete = data.get("athlete", {})
    activities = data.get("activities", [])
    pmc = data.get("pmc", [])

    lines = []
    lines.append(_fmt_header(
        f"🏋️  {template_type.upper()} RECOMMENDATION"
    ))
    lines.append(f"Athlete: {athlete.get('name', '?')}")
    lines.append(f"FTP: {athlete.get('ftp', '?')}W")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Activities in range: {len(activities)}")
    lines.append("")

    # PMC summary
    if pmc:
        latest = pmc[-1]
        lines.append(_fmt_section("TRAINING LOAD (Performance Management Chart)", ""))
        lines.append(f"  CTL (Fitness):     {latest.get('fitness_ctl', '?'):.1f}")
        lines.append(f"  ATL (Fatigue):     {latest.get('fatigue_atl', '?'):.1f}")
        lines.append(f"  TSB (Form):        {latest.get('form_tsb', '?'):.1f}")

        # Trend
        if len(pmc) >= 2:
            prev = pmc[-2]
            tsb_trend = latest.get('form_tsb', 0) - prev.get('form_tsb', 0)
            ctl_trend = latest.get('fitness_ctl', 0) - prev.get('fitness_ctl', 0)
            lines.append(f"  TSB Trend:         {'+' if tsb_trend >= 0 else ''}{tsb_trend:.1f} (last week)")
            lines.append(f"  CTL Trend:         {'+' if ctl_trend >= 0 else ''}{ctl_trend:.1f} (last week)")

        # Reading
        tsb = latest.get('form_tsb', 0)
        if tsb < -20:
            form_read = "🔴 Deep fatigue — prioritise rest and easy spinning"
        elif tsb < -10:
            form_read = "🟡 Accumulated fatigue — consider an easier day"
        elif tsb < 5:
            form_read = "🟢 Normal training range — good to train"
        elif tsb < 15:
            form_read = "🟢 Fresh — good form, ideal for quality work"
        else:
            form_read = "🟢 Very fresh — possibly detraining, build volume gradually"
        lines.append(f"  Form Reading:      {form_read}")

    lines.append("")

    # Recent activities
    if activities:
        lines.append(_fmt_section("RECENT ACTIVITIES (last 14 days)", ""))
        # Sort by date desc
        sorted_acts = sorted(
            activities,
            key=lambda a: a.get("start_date", ""),
            reverse=True,
        )
        for a in sorted_acts[:10]:
            name = a.get("name", "Activity?")
            date_str = (a.get("start_date") or "?")[:10]
            tss = a.get("tss", "?")
            dist = a.get("distance_km", 0)
            time_m = (a.get("moving_time_seconds", 0) or 0) // 60
            np = a.get("weighted_avg_watts", "")
            wtype = a.get("classification", {}).get("workout_type_label", "")
            lines.append(
                f"  {date_str} | {name[:40]:40s} | "
                f"{f'TSS:{int(tss)}' if isinstance(tss, (int, float)) and tss else '':>10s} | "
                f"{f'{dist:.0f}km' if dist else '':>8s} | "
                f"{time_m}min | {wtype}"
            )

    lines.append("")

    # Workout suggestion
    if template_type == "daily":
        lines.append(_fmt_section("SUGGESTED WORKOUT TYPE", ""))

        tsb_val = pmc[-1].get("form_tsb", 0) if pmc else 0

        if tsb_val < -15:
            lines.append("  Recovery / Easy Spin (Z1-Z2, 30-45min)")
        elif tsb_val < -5:
            lines.append("  Endurance (Z2, 60-90min)")
        elif tsb_val < 5:
            lines.append("  Tempo or Sweet Spot (60-75min)")
        elif tsb_val < 15:
            lines.append("  Sweet Spot or Threshold (60-75min)")
        else:
            lines.append("  Build endurance or VO2max work (60-90min)")

        lines.append("")
        lines.append(_fmt_section("READY TO GENERATE", ""))
        lines.append("Call the LLM or pass this context to generate a full workout.")

    elif template_type == "weekly":
        lines.append(_fmt_section("WEEKLY TRAINING SUMMARY", ""))
        week_tss = sum(a.get("tss", 0) or 0 for a in activities)
        week_dist = sum(a.get("distance_km", 0) or 0 for a in activities)
        week_rides = len(activities)
        lines.append(f"  Total rides:       {week_rides}")
        lines.append(f"  Total TSS:         {week_tss:.0f}")
        lines.append(f"  Total distance:    {week_dist:.1f} km")

        # Balance
        if pmc:
            atl = pmc[-1].get("fatigue_atl", 0)
            ctl = pmc[-1].get("fitness_ctl", 0)
            chronic_load = ctl * 7  # approximate weekly
            lines.append(f"  Current weekly load: ~{chronic_load:.0f} TSS")
            ratio = atl / ctl if ctl > 0 else 0
            lines.append(f"  Acute/Chronic ratio: {ratio:.2f} "
                         f"{'(fatigue building)' if ratio > 1.3 else '(balanced)' if ratio > 0.8 else '(low stimulus)'}")

    elif template_type == "assessment":
        lines.append(_fmt_section("FORM ASSESSMENT", ""))
        if pmc:
            latest = pmc[-1]
            tsb = latest.get("form_tsb", 0)
            if tsb < -20:
                lines.append("  ⚠️ You're in a deep fatigue hole. Take 2-3 easy days.")
            elif tsb < -10:
                lines.append("  ⚠️ Fatigue is building. Consider a rest day or recovery ride.")
            elif tsb < 5:
                lines.append("  ✅ In the training zone. You're ready to train.")
            elif tsb < 15:
                lines.append("  ✅ Fresh legs. Great time for quality work.")
            else:
                lines.append("  ⚠️ Very fresh — you may be losing fitness. Start building volume.")

            lines.append("")
            lines.append("  Recommendation based on your current form:")

            if tsb < -10:
                lines.append("  • Take today easy (Z1-Z2, 30-45min)")
                lines.append("  • Focus on sleep and nutrition")
                lines.append("  • Skip hard efforts until TSB recovers above -10")
            elif tsb < 5:
                lines.append("  • Normal training day")
                lines.append("  • Tempo or Sweet Spot session")
                lines.append("  • Duration: 60-75min")
            elif tsb < 15:
                lines.append("  • Quality session — threshold or sweet spot")
                lines.append("  • Can push intensity today")
                lines.append("  • Duration: 60-75min with warm-up")
            else:
                lines.append("  • Build endurance (Z2, 90-120min)")
                lines.append("  • Consider adding some Z3/Z4 work to stimulate fitness gains")

    return "\n".join(lines)


def _build_coaching_context(data: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact coaching context dict for LLM consumption."""
    athlete = data.get("athlete", {})
    activities = data.get("activities", [])
    pmc = data.get("pmc", [])

    context = {
        "athlete": {
            "name": athlete.get("name"),
            "ftp": athlete.get("ftp"),
            "weight_kg": athlete.get("weight_kg"),
            "recent_ftp": athlete.get("estimated_ftp"),
        },
        "training_load": {},
        "weekly_summary": {},
        "recent_activities": [],
    }

    if pmc:
        latest = pmc[-1]
        context["training_load"] = {
            "ctl": round(latest.get("fitness_ctl", 0), 1),
            "atl": round(latest.get("fatigue_atl", 0), 1),
            "tsb": round(latest.get("form_tsb", 0), 1),
        }

    # Weekly stats
    week_tss = sum(a.get("tss", 0) or 0 for a in activities)
    week_dist = sum(a.get("distance_km", 0) or 0 for a in activities)
    week_rides = len(activities)
    context["weekly_summary"] = {
        "rides": week_rides,
        "total_tss": round(week_tss, 0),
        "distance_km": round(week_dist, 1),
    }

    # Recent activities (top 5)
    sorted_acts = sorted(
        activities,
        key=lambda a: a.get("start_date", ""),
        reverse=True,
    )
    for a in sorted_acts[:5]:
        context["recent_activities"].append({
            "date": (a.get("start_date") or "")[:10],
            "name": a.get("name", ""),
            "type": a.get("classification", {}).get("workout_type_label", ""),
            "tss": a.get("tss"),
            "duration_min": (a.get("moving_time_seconds", 0) or 0) // 60,
            "distance_km": a.get("distance_km"),
            "np": a.get("weighted_avg_watts"),
        })

    return context


# ── CLI ──


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Get fitness recommendations for a specific Discord user",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.query_for_user joey0624                # Today's recommendation
  python -m app.query_for_user joey0624 --daily         # Explicit daily
  python -m app.query_for_user joey0624 --weekly        # Weekly plan
  python -m app.query_for_user joey0624 --assessment    # Form assessment
  python -m app.query_for_user joey0624 --context       # JSON context
  python -m app.query_for_user --list-users             # List registered users
        """,
    )

    parser.add_argument("discord_id", nargs="?", help="Discord username (not needed with --list-users)")
    parser.add_argument("--daily", action="store_true", help="Daily workout recommendation (default)")
    parser.add_argument("--weekly", action="store_true", help="Weekly training plan")
    parser.add_argument("--assessment", action="store_true", help="Form/fatigue assessment")
    parser.add_argument("--context", action="store_true", help="Output structured JSON context")
    parser.add_argument("--days", type=int, default=14, help="Days of training history")
    parser.add_argument("--list-users", action="store_true", help="List registered users")
    parser.add_argument("--show-key", action="store_true",
                        help="Show decrypted API key (debugging only)")

    args = parser.parse_args()

    # ── List users ──
    if args.list_users:
        users = list_discord_users()
        if not users:
            print("No Discord users registered.")
            print("Register one with:")
            print("  python -m app.user_manager register <discord_id> --intervals-key ... --athlete-id ...")
            return
        print(f"{'#':>3} {'Discord ID':20s} {'Name':20s} {'Athlete ID':15s} {'FTP':>5} {'Days':>5}")
        print("-" * 68)
        for u in users:
            print(f"{u['id']:3d} {u['discord_user_id']:20s} {(u['name'] or ''):20s} "
                  f"{(u['athlete_id'] or ''):15s} {u['ftp']:5d} {'✓' if u['has_api_key'] else '✗':>5}")
        return

    if not args.discord_id:
        parser.print_help()
        sys.exit(1)

    # ── Resolve template type ──
    if args.weekly:
        template_type = "weekly"
    elif args.assessment:
        template_type = "assessment"
    else:
        template_type = "daily"

    # ── Show key (debug) ──
    if args.show_key:
        user = get_user_by_discord(args.discord_id)
        if user and user.get("intervals_api_key"):
            print(f"🔑 API key for {args.discord_id}: {user['intervals_api_key']}")
        else:
            print(f"No API key found for {args.discord_id}")
        return

    # ── Fetch and process ──
    try:
        data = get_training_data_for_user(args.discord_id, days_back=args.days)
    except (ValueError, RuntimeError) as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    athlete = data.get("athlete", {})
    print(f"📡 Fetched data for {athlete.get('name', '?')} (FTP: {athlete.get('ftp', '?')}W)", file=sys.stderr)
    print(file=sys.stderr)

    if args.context:
        # JSON context output
        ctx = _build_coaching_context(data)
        print(json.dumps(ctx, indent=2))
    else:
        # Human-readable recommendation
        rec = _format_recommendation(data, template_type=template_type)
        print(rec)

    # Also print a JSON context block at the bottom for the agent to use
    print(file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print("LLM CONTEXT (JSON — pass to LLM for final generation)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    ctx = _build_coaching_context(data)
    print(json.dumps(ctx, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
