"""
Per-User Training Query — get LLM-powered fitness advice for any registered Discord user.

Fetches training data from Intervals.icu, builds a structured
context pack with all training metrics, and generates an LLM-powered recommendation.

The context JSON is the primary output — the Hermes agent can feed it to any LLM.
As a convenience, the tool also supports direct LLM generation via an integrated
OpenAI-compatible API call (configured via env vars or platform defaults).

Usage (for the Hermes agent):
    python -m app.query_for_user joey0624                        # LLM recommendation (default)
    python -m app.query_for_user joey0624 --context              # Structured JSON (for agent LLM)
    python -m app.query_for_user joey0624 --rule-based           # Rule-based fallback
    python -m app.query_for_user joey0624 --weekly               # Weekly plan (LLM)
    python -m app.query_for_user joey0624 --assessment           # Form assessment (LLM)
    python -m app.query_for_user joey0624 --query "How was my week?"  # NL query
    python -m app.query_for_user --list-users                    # List registered users

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

# ── LLM Integration ──

DEFAULT_LLM_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_LLM_MODEL = "deepseek-v4-flash"


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
) -> str:
    """Call an OpenAI-compatible chat completion API."""
    import httpx
    import os

    url = (base_url or os.environ.get("LLM_BASE_URL") or DEFAULT_LLM_BASE_URL).rstrip("/")
    key = api_key or os.environ.get("LLM_API_KEY") or ""
    model_name = model or os.environ.get("LLM_MODEL") or DEFAULT_LLM_MODEL

    if not url.endswith("/chat/completions"):
        url = url.rstrip("/") + "/chat/completions"

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 2048,
    }

    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")


def _build_llm_messages(data: Dict[str, Any], template_type: str) -> tuple[str, str]:
    """Build system + user prompts for LLM from training data context.

    Constructs a coaching prompt using the athlete's actual Intervals.icu
    training data (PMC, weekly summary, recent activities).
    """
    athlete = data.get("athlete", {})
    pmc = data.get("pmc", [])
    activities = data.get("activities", [])
    weekly = data.get("weekly_summary", {})

    ftp = athlete.get("ftp") or 0
    weight = athlete.get("weight_kg") or 0
    wpk = round(ftp / weight, 2) if ftp and weight else 0

    # Build data context string
    ctx_lines = []
    ctx_lines.append(f"Athlete: {athlete.get('name', '?')} | FTP: {ftp}W | W/kg: {wpk}")

    if pmc:
        latest = pmc[-1]
        ctx_lines.append(f"CTL (fitness): {latest['fitness_ctl']:.0f}")
        ctx_lines.append(f"ATL (fatigue): {latest['fatigue_atl']:.0f}")
        ctx_lines.append(f"TSB (form): {latest['form_tsb']:+.0f}")
        if latest['form_tsb'] < -20:
            ctx_lines.append("Form reading: Deep fatigue")
        elif latest['form_tsb'] < -10:
            ctx_lines.append("Form reading: Accumulated fatigue")
        elif latest['form_tsb'] < 5:
            ctx_lines.append("Form reading: Normal training range")
        elif latest['form_tsb'] < 15:
            ctx_lines.append("Form reading: Fresh — ideal for quality")
        else:
            ctx_lines.append("Form reading: Very fresh — possible detraining")

    ctx_lines.append(f"Weekly: {weekly.get('ride_count', 0)} rides, "
                     f"{weekly.get('total_tss', 0)} TSS, "
                     f"{weekly.get('total_distance_km', 0):.0f} km")

    if activities:
        ctx_lines.append("\nRecent activities (last 14 days):")
        sorted_acts = sorted(activities, key=lambda a: a.get("start_date", ""), reverse=True)
        for a in sorted_acts[:5]:
            ctx_lines.append(
                f"  {(a.get('start_date') or '?')[:10]} | {a.get('name', '?')[:30]} | "
                f"TSS:{a.get('tss', '?')} | "
                f"NP:{a.get('weighted_avg_watts', '?')}W | "
                f"{a.get('classification', {}).get('workout_type_label', '')}"
            )

    context_str = "\n".join(ctx_lines)

    system_prompt = (
        "You are an expert cycling coach. You analyse training data from Intervals.icu "
        "and provide personalised training recommendations. "
        "Always express power targets as percentages of FTP, never raw watts. "
        "Be specific, actionable, and encouraging. "
        "Format your response cleanly with markdown."
    )

    type_labels = {
        "daily": "today's workout recommendation",
        "weekly": "a weekly training plan",
        "assessment": "a form/fatigue assessment",
    }
    type_label = type_labels.get(template_type, "a training recommendation")

    user_prompt = (
        f"Generate {type_label} based on this athlete's training data from Intervals.icu:\n\n"
        f"{context_str}\n\n"
        "Provide your recommendation in clear markdown. "
        "If suggesting power or pace targets, use % of FTP."
    )

    return system_prompt, user_prompt


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
        epilog="""Examples:
  python -m app.query_for_user joey0624                # LLM-powered recommendation (default)
  python -m app.query_for_user joey0624 --daily         # Explicit daily (LLM-powered)
  python -m app.query_for_user joey0624 --rule-based    # Rule-based fallback
  python -m app.query_for_user joey0624 --weekly        # Weekly plan (LLM-powered)
  python -m app.query_for_user joey0624 --assessment    # Form assessment (LLM-powered)
  python -m app.query_for_user joey0624 --context       # JSON context (for agent to feed LLM)
  python -m app.query_for_user --list-users             # List registered users
        """,
    )

    parser.add_argument("discord_id", nargs="?", help="Discord username (not needed with --list-users)")
    parser.add_argument("--daily", action="store_true", help="Daily workout recommendation (LLM-powered, default)")
    parser.add_argument("--weekly", action="store_true", help="Weekly training plan (LLM-powered)")
    parser.add_argument("--assessment", action="store_true", help="Form/fatigue assessment (LLM-powered)")
    parser.add_argument("--context", action="store_true", help="Output structured JSON context (for agent LLM)")
    parser.add_argument("--rule-based", action="store_true", help="Use rule-based fallback instead of LLM")
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
        # JSON context output — for Hermes agent to feed to its own LLM
        ctx = _build_coaching_context(data)
        print(json.dumps(ctx, indent=2))
    elif args.rule_based:
        # Rule-based fallback
        rec = _format_recommendation(data, template_type=template_type)
        print(rec)
    else:
        # LLM-powered recommendation (DEFAULT) — Intervals.icu data in, LLM recommendation out
        print("🤖 Generating LLM recommendation from training data...", file=sys.stderr)
        try:
            system_prompt, user_prompt = _build_llm_messages(data, template_type)
            rec = _call_llm(system_prompt, user_prompt)
            print(rec)
        except Exception as e:
            print(f"⚠️ LLM call failed ({e}), falling back to rule-based", file=sys.stderr)
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
