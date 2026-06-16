"""
Workout prescription CLI — one-command access to AI-powered coaching.

Fetches latest training data from Intervals.icu, builds a structured
coaching prompt using the template engine, and outputs a workout
recommendation. Optionally calls an LLM endpoint for generation.

Usage:
    python -m app.prescribe                   # Daily workout (default)
    python -m app.prescribe --weekly          # Weekly training plan
    python -m app.prescribe --assessment      # Form/fatigue assessment
    python -m app.prescribe --llm             # Generate with LLM

Environment:
    INTERVALS_API_KEY    Required. Intervals.icu API key
    INTERVALS_ATHLETE_ID Required. Intervals.icu athlete ID
    LLM_API_KEY          Optional. API key for OpenAI-compatible endpoint
    LLM_BASE_URL         Optional. API base URL (default: opencode endpoint)
    LLM_MODEL            Optional. Model name (default: deepseek-v4-flash)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default LLM endpoint — can be overridden via env
DEFAULT_LLM_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_LLM_MODEL = "deepseek-v4-flash"

# ── Template registry ──

TEMPLATE_NAMES = {
    "daily": "daily_workout_prompt",
    "weekly": "weekly_plan_prompt",
    "assessment": "form_assessment_prompt",
    "periodization": "periodization_prompt",
}


def _get_template_builder(name: str):
    """Lazy-import and return the named template builder."""
    from app.prompts.coaching_templates import get_template
    return get_template(name)


# ── Data fetching ──


def _fetch_data(days_back: int = 42) -> Dict[str, Any]:
    """Fetch and return training data."""
    from app.data_fetcher import TrainingDataFetcher
    fetcher = TrainingDataFetcher()
    return fetcher.fetch_all(days_back=days_back)


# ── LLM caller ──


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

    url = (base_url or os.environ.get("LLM_BASE_URL") or DEFAULT_LLM_BASE_URL).rstrip("/")
    key = api_key or os.environ.get("LLM_API_KEY") or ""
    model_name = model or os.environ.get("LLM_MODEL") or DEFAULT_LLM_MODEL

    # Ensure we have the /chat/completions path
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


# ── Output formatters ──


def _fmt_header(title: str) -> str:
    """Print a nice header for CLI output."""
    sep = "=" * 60
    return f"{sep}\n{title}\n{sep}"


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(
        description="Generate cycling workout prescriptions from training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.prescribe                         # Daily workout prompt
  python -m app.prescribe --weekly                 # Weekly plan prompt
  python -m app.prescribe --assessment             # Form assessment prompt
  python -m app.prescribe --llm                    # Generate with LLM
  python -m app.prescribe --periodization "Gran Fondo" --target-date 2026-08-15
        """,
    )

    # Template selection
    parser.add_argument("--daily", action="store_true", help="Daily workout prescription (default)")
    parser.add_argument("--weekly", action="store_true", help="Weekly training plan")
    parser.add_argument("--assessment", action="store_true", help="Form/fatigue assessment")
    parser.add_argument("--periodization", type=str, metavar="EVENT",
                        help="Periodization plan for a target event (e.g. 'Gran Fondo')")
    parser.add_argument("--target-date", type=str, default="",
                        help="Target date for periodization (YYYY-MM-DD)")

    # LLM options
    parser.add_argument("--llm", action="store_true", help="Generate with LLM (requires API key)")
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    parser.add_argument("--api-key", type=str, default=None, help="LLM API key")
    parser.add_argument("--base-url", type=str, default=None, help="LLM base URL")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature")

    # Data options
    parser.add_argument("--days", type=int, default=42, help="Days of training history")
    parser.add_argument("--show-data", action="store_true", help="Show the context data being used")

    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    # ── Resolve template ──
    if args.periodization:
        template_name = "periodization"
    elif args.weekly:
        template_name = "weekly"
    elif args.assessment:
        template_name = "assessment"
    else:
        template_name = "daily"

    # ── Fetch data ──
    print(f"📡 Fetching {args.days} days of training data from Intervals.icu...", file=sys.stderr)
    try:
        data = _fetch_data(days_back=args.days)
    except (ValueError, PermissionError, RuntimeError) as e:
        print(f"❌ Failed to fetch data: {e}", file=sys.stderr)
        sys.exit(1)

    athlete = data.get("athlete", {})
    print(f"✅ Loaded data for {athlete.get('name', '?')} (FTP: {athlete.get('ftp', '?')}W)", file=sys.stderr)

    # ── Show data if requested ──
    if args.show_data:
        from app.context_pack import build_context_pack
        ctx = build_context_pack(data, compact=False)
        print(f"\n{_fmt_header('TRAINING DATA CONTEXT')}\n", file=sys.stderr)
        print(ctx, file=sys.stderr)
        print(file=sys.stderr)

    # ── Build prompt ──
    builder = _get_template_builder(template_name)
    if template_name == "periodization":
        target_event = args.periodization or "target event"
        target_date = args.target_date or "2026-08-01"
        prompts = builder(data, target_event=target_event, target_date=target_date)
    else:
        prompts = builder(data)

    # ── Generate or output ──
    if args.llm:
        print(f"🤖 Generating with LLM...", file=sys.stderr)
        try:
            response = _call_llm(
                system_prompt=prompts["system"],
                user_prompt=prompts["user"],
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
            )
            print(response)
        except RuntimeError as e:
            print(f"❌ LLM generation failed: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Just output the assembled prompt
        print(prompts["user"])


if __name__ == "__main__":
    main()
