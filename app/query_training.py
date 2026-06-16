"""
Natural Language Training Queries — translates plain English questions
about training data into structured LLM prompts with data context.

Examples:
  "How was my training this week?"
  "What's my form trend?"
  "Compare this week to last week"
  "Am I overtraining?"
  "What should I do tomorrow?"
  "How's my FTP progression?"

Usage (CLI):
    python -m app.query_training "How was my training this week?"

Usage (API):
    POST /api/coaching/query  {"query": "How was my training this week?"}
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Query classifier ──


def classify_query(query: str) -> str:
    """Classify a natural language query into a training question type."""
    q = query.lower().strip()

    # Weekly review
    if any(w in q for w in ["this week", "this past week", "last week", "weekly review",
                             "how did i do this week", "week in review"]):
        return "weekly_review"

    # Form / fatigue
    if any(w in q for w in ["form", "fatigue", "tsb", "overtraining", "overtrain",
                            "burnout", "am i fresh", "am i tired", "readiness"]):
        return "form_assessment"

    # Compare
    if any(w in q for w in ["compare", "vs", "versus", "better than", "difference",
                            "how does this week compare"]):
        return "comparison"

    # What to do
    if any(w in q for w in ["what should i do", "what's next", "todays workout",
                            "what workout", "suggest", "recommend", "train today",
                            "what to do", "plan for"]):
        return "workout_suggestion"

    # FTP / progression
    if any(w in q for w in ["ftp", "progress", "getting faster", "improving",
                            "power", "getting stronger", "am i improving"]):
        return "progression"

    # General analysis
    if any(w in q for w in ["analysis", "analyze", "how am i doing", "status",
                            "summary", "overview", "dashboard"]):
        return "general_analysis"

    # Trends / patterns
    if any(w in q for w in ["trend", "pattern", "noticing", "changes",
                            "volume", "consistency"]):
        return "trends"

    return "general_analysis"


# ── Data collectors ──


def _collect_weekly_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Collect relevant data for a weekly review."""
    activities = data.get("activities", [])
    weekly = data.get("weekly_summary", {})
    pmc = data.get("pmc", [])
    athlete = data.get("athlete", {})

    # Last 7 days
    today = date.today()
    week_ago = today - timedelta(days=7)
    week_acts = [a for a in activities if (a.get("start_date") or "")[:10] >= week_ago.isoformat()]

    # Last 14 days
    two_weeks_ago = today - timedelta(days=14)
    prev_week_acts = [
        a for a in activities
        if two_weeks_ago.isoformat() <= (a.get("start_date") or "")[:10] < week_ago.isoformat()
    ]

    # Breakdown by type
    type_breakdown = {}
    for a in week_acts:
        wtype = a.get("classification", {}).get("workout_type_label", "Unknown")
        type_breakdown[wtype] = type_breakdown.get(wtype, 0) + 1

    # Weekly tally
    this_week_tss = sum(a.get("tss", 0) or 0 for a in week_acts)
    prev_week_tss = sum(a.get("tss", 0) or 0 for a in prev_week_acts)
    this_week_dist = sum(a.get("distance_km", 0) or 0 for a in week_acts)
    this_week_time = sum(a.get("moving_time_seconds", 0) or 0 for a in week_acts) // 60

    return {
        "date_range": f"{week_ago.strftime('%b %d')} - {today.strftime('%b %d')}",
        "total_rides": len(week_acts),
        "total_tss": this_week_tss,
        "prev_week_tss": prev_week_tss,
        "tss_change": this_week_tss - prev_week_tss,
        "total_distance_km": round(this_week_dist, 1),
        "total_time_minutes": this_week_time,
        "average_tss_per_ride": round(this_week_tss / len(week_acts), 0) if week_acts else 0,
        "type_breakdown": type_breakdown,
        "form": pmc[-1]["form_tsb"] if pmc else None,
        "fitness_ctl": pmc[-1]["fitness_ctl"] if pmc else None,
        "fatigue_atl": pmc[-1]["fatigue_atl"] if pmc else None,
    }


def _collect_form_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Collect data for form/fatigue assessment."""
    pmc = data.get("pmc", [])
    activities = data.get("activities", [])

    form_data = {}
    if pmc:
        latest = pmc[-1]
        form_data = {
            "current_form_tsb": latest.get("form_tsb"),
            "fitness_ctl": latest.get("fitness_ctl"),
            "fatigue_atl": latest.get("fatigue_atl"),
        }

        if len(pmc) >= 2:
            prev = pmc[-2]
            form_data["form_trend"] = round(latest["form_tsb"] - prev["form_tsb"], 1)
            form_data["ctl_trend"] = round(latest["fitness_ctl"] - prev["fitness_ctl"], 1)

    # Rest days
    today = date.today()
    last_5_days = [(today - timedelta(days=i)).isoformat() for i in range(5)]
    act_dates = set((a.get("start_date") or "")[:10] for a in activities)
    rest_days = sum(1 for d in last_5_days if d not in act_dates)

    form_data["rest_days_last_5"] = rest_days
    return form_data


def _collect_comparison_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Collect data for week-over-week comparison."""
    today = date.today()
    this_wk = _collect_weekly_data(data)

    two_weeks_ago = today - timedelta(days=14)
    three_weeks_ago = today - timedelta(days=21)

    activities = data.get("activities", [])
    prev_acts = [a for a in activities if two_weeks_ago.isoformat() <= (a.get("start_date") or "")[:10] < (today - timedelta(days=7)).isoformat()]

    # Look for a third week
    wk3_acts = [a for a in activities if three_weeks_ago.isoformat() <= (a.get("start_date") or "")[:10] < two_weeks_ago.isoformat()]

    return {
        "this_week": {
            "tss": this_wk["total_tss"],
            "rides": this_wk["total_rides"],
            "distance_km": this_wk["total_distance_km"],
            "time_min": this_wk["total_time_minutes"],
        },
        "last_week": {
            "tss": this_wk["prev_week_tss"],
            "rides": len(prev_acts),
            "distance_km": round(sum(a.get("distance_km", 0) or 0 for a in prev_acts), 1),
            "time_min": sum(a.get("moving_time_seconds", 0) or 0 for a in prev_acts) // 60,
        },
        "three_weeks_ago": {
            "tss": sum(a.get("tss", 0) or 0 for a in wk3_acts),
            "rides": len(wk3_acts),
        },
    }


def _collect_progression_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Collect data for FTP/progression analysis."""
    activities = data.get("activities", [])
    athlete = data.get("athlete", {})

    # Get last 3 months of activities
    recent = sorted(
        [a for a in activities if (a.get("start_date") or "") >= (date.today() - timedelta(days=90)).isoformat()],
        key=lambda a: a.get("start_date", ""),
    )

    # Monthly TSS
    monthly_tss = {}
    for a in recent:
        month = (a.get("start_date") or "")[:7]
        monthly_tss[month] = monthly_tss.get(month, 0) + (a.get("tss", 0) or 0)

    # Monthly volume
    monthly_volume = {}
    for a in recent:
        month = (a.get("start_date") or "")[:7]
        monthly_volume[month] = monthly_volume.get(month, 0) + (a.get("distance_km", 0) or 0)

    return {
        "current_ftp": athlete.get("ftp"),
        "months_of_data": sorted(monthly_tss.keys()),
        "monthly_tss": monthly_tss,
        "monthly_volume_km": monthly_volume,
        "total_activities_90d": len(recent),
    }


# ── Query context builder ──


def build_query_context(data: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Build a structured context dict for an NL query.

    Returns the context + the classified query type + a formatted prompt.
    """
    qtype = classify_query(query)

    collectors = {
        "weekly_review": _collect_weekly_data,
        "form_assessment": _collect_form_data,
        "comparison": _collect_comparison_data,
        "workout_suggestion": _collect_weekly_data,
        "progression": _collect_progression_data,
        "general_analysis": _collect_weekly_data,
        "trends": _collect_progression_data,
    }

    collector = collectors.get(qtype, _collect_weekly_data)
    query_data = collector(data)

    return {
        "query": query,
        "query_type": qtype,
        "context_data": query_data,
        "athlete_name": data.get("athlete", {}).get("name", "Rider"),
        "fetched_at": data.get("fetched_at"),
    }


QUERY_SYSTEM_PROMPT = """You are a professional cycling coach's data analyst. You interpret training data 
and answer questions about it in clear, actionable language. You have access to the rider's 
recent training data, PMC metrics, and activity history.

Rules:
- Be specific with numbers (TSS, hours, km, form score)
- Explain trends and patterns, not just raw data
- Flag risks (overtraining, stale form, insufficient recovery)
- Suggest actionable next steps when appropriate
- Keep responses concise and focused on what the rider asked
- If asked about today's workout, include a specific recommendation
"""


def build_query_prompt(context: Dict[str, Any]) -> Dict[str, str]:
    """Build a system + user prompt pair for the natural language query."""
    user_prompt = f"""The rider asks: "{context['query']}"

This is classified as a {context['query_type'].replace('_', ' ')} type question.

Here is their training context:

```json
{json.dumps(context['context_data'], indent=2, default=str)}
```

Answer their question based on this data. Be specific and actionable."""

    return {
        "system": QUERY_SYSTEM_PROMPT,
        "user": user_prompt,
    }


# ── CLI ──


def main():
    """CLI entrypoint for natural language queries."""
    import argparse

    parser = argparse.ArgumentParser(description="Query training data in natural language")
    parser.add_argument("query", nargs="*", help="Natural language query")
    parser.add_argument("--days", type=int, default=42, help="Days of history")
    parser.add_argument("--show-context", action="store_true", help="Show the raw context data")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    query = " ".join(args.query) if args.query else "How was my training this week?"

    try:
        from app.data_fetcher import TrainingDataFetcher
        fetcher = TrainingDataFetcher()
        data = fetcher.fetch_all(days_back=args.days)
    except (ImportError, ValueError, RuntimeError) as e:
        print(f"ERROR: Failed to fetch data: {e}", file=sys.stderr)
        sys.exit(1)

    context = build_query_context(data, query)
    prompts = build_query_prompt(context)

    if args.show_context:
        print(json.dumps(context["context_data"], indent=2, default=str))

    if args.json:
        print(json.dumps({
            "query_type": context["query_type"],
            "prompts": prompts,
        }, indent=2))
    else:
        print(f"\nQuery: {query}")
        print(f"Type: {context['query_type']}")
        print()
        print("=== SYSTEM PROMPT ===")
        print(prompts["system"])
        print()
        print("=== USER PROMPT ===")
        print(prompts["user"])


if __name__ == "__main__":
    main()
