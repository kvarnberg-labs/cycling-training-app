"""
Intervals.icu sync — pull training data into local database.

Always run this before querying a user's data. It:
  1. Fetches latest activities from Intervals.icu API
  2. Fetches training metrics (CTL/ATL/TSB)
  3. Updates user profile (FTP, weight, resting HR, max HR)
  4. Stores everything in the local database

Usage:
    python -m app.sync_intervals <discord_id>
        Syncs all data for a registered Discord user.

    python -m app.sync_intervals <discord_id> --days 14
        Sync only the last 14 days.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import StravaActivity, TrainingMetrics, User
from app.services.encryption import decrypt
from app.services.intervals_client import IntervalsClient, IntervalsError
from app.user_manager import get_user_by_discord

logger = logging.getLogger(__name__)


# ── Helpers ──


def _activity_to_dict(act: Dict[str, Any]) -> Dict[str, Any]:
    """Map Intervals.icu activity fields to local model fields."""
    return {
        "strava_id": act.get("strava_id") or abs(hash(str(act.get("id", "")))),
        "name": act.get("name", "Untitled"),
        "activity_type": act.get("type") or act.get("sport") or "Ride",
        "start_date": act.get("start_date") or act.get("start_date_local"),
        "timezone": act.get("timezone"),
        "elapsed_time": act.get("elapsed_time") or act.get("elapsedTime", 0),
        "moving_time": act.get("moving_time") or act.get("movingTime", 0),
        "distance": act.get("distance", 0),
        "total_elevation_gain": act.get("total_elevation_gain") or act.get("elevationGain", 0),
        "average_watts": act.get("average_watts") or act.get("avgPower") or act.get("icu_average_watts"),
        "max_watts": act.get("max_watts") or act.get("maxPower") or act.get("icu_max_watts"),
        "weighted_average_watts": act.get("weighted_average_watts") or act.get("normalizedPower") or act.get("icu_weighted_avg_watts"),
        "average_heartrate": act.get("average_heartrate") or act.get("avgHeartRate"),
        "max_heartrate": act.get("max_heartrate") or act.get("maxHeartRate"),
        "average_cadence": act.get("average_cadence") or act.get("avgCadence"),
        "kilojoules": act.get("kilojoules") or act.get("kJ", 0),
        "intensity_factor": act.get("intensity_factor") or act.get("intensityFactor", 0),
        "training_stress_score": act.get("training_stress_score") or act.get("tss", 0),
        "training_load": act.get("training_stress_score") or act.get("tss", 0),
        "workout_type": act.get("workout_type") or act.get("sub_type"),
    }


def _training_metrics_to_dict(tm: Dict[str, Any]) -> Dict[str, Any]:
    """Map athlete-summary entry to local TrainingMetrics fields."""
    return {
        "date": tm.get("date"),
        "ctl": tm.get("fitness", 0),
        "atl": tm.get("fatigue", 0),
        "tsb": tm.get("form", 0),
        "total_tss": tm.get("training_load", 0),
        "total_duration_minutes": tm.get("time", 0) // 60 if tm.get("time") else 0,
        "total_distance_km": (tm.get("distance", 0) or 0) / 1000,
        "total_kj": tm.get("kj", 0),
        "ride_count": tm.get("rides", 0),
    }


# ── Sync ──


def sync_user(discord_id: str, days_back: int = 90) -> dict:
    """Sync a Discord user's data from Intervals.icu into the local database.

    Args:
        discord_id: Discord username
        days_back: How many days of history to sync

    Returns:
        Dict with sync results: synced_activities, synced_metrics, profile_updates
    """
    import asyncio

    user_info = get_user_by_discord(discord_id)
    if not user_info:
        raise ValueError(f"No user registered with Discord ID '{discord_id}'.")

    api_key = user_info.get("intervals_api_key")
    athlete_id = user_info.get("intervals_athlete_id")
    if not api_key or not athlete_id:
        raise ValueError(f"User '{discord_id}' has no Intervals.icu credentials.")

    db: Session = SessionLocal()
    try:
        user = db.query(User).filter(User.discord_user_id == discord_id).first()
        if not user:
            raise ValueError(f"User '{discord_id}' not found in database.")

        result = asyncio.run(_do_sync(user, api_key, athlete_id, days_back, db))
        db.commit()
        return result
    finally:
        db.close()


async def _do_sync(
    user: User,
    api_key: str,
    athlete_id: str,
    days_back: int,
    db: Session,
) -> dict:
    """Perform the actual sync operations."""
    client = IntervalsClient(api_key=api_key, athlete_id=athlete_id)
    result = {"synced_activities": 0, "synced_metrics": 0, "profile_updates": {}}

    # ── Step 1: Update athlete profile from Intervals.icu ──
    try:
        athlete = await client.get_athlete()
        updates = {}

        # Extract FTP from sportSettings
        sport_settings = athlete.get("sportSettings", [])
        for ss in sport_settings:
            if ss.get("types") and "Ride" in ss.get("types", []):
                ftp = ss.get("ftp")
                if ftp and ftp != user.ftp:
                    updates["ftp"] = ftp
                break

        # Extract weight
        weight = athlete.get("icu_weight") or athlete.get("weight")
        if weight and weight != user.weight_kg:
            updates["weight_kg"] = weight

        # Extract resting HR
        rhr = athlete.get("icu_resting_hr")
        if rhr and rhr != user.resting_hr:
            updates["resting_hr"] = rhr

        # Extract max HR
        mhr = athlete.get("athlete_max_hr")
        if mhr and mhr != user.max_hr:
            updates["max_hr"] = mhr

        if updates:
            for key, value in updates.items():
                setattr(user, key, value)
            result["profile_updates"] = updates
            print(f"✅ Updated profile: {updates}", file=sys.stderr)

    except Exception as e:
        print(f"⚠️ Profile update skipped: {e}", file=sys.stderr)

    # ── Step 2: Sync activities ──
    try:
        activities = await client.get_activities(days_back=days_back, limit=200)
    except IntervalsError as e:
        print(f"❌ Failed to fetch activities: {e}", file=sys.stderr)
        return result

    if not activities:
        print("ℹ️  No activities found in Intervals.icu", file=sys.stderr)
        return result

    # Deduplicate against existing local records
    existing_ids = set(
        row[0]
        for row in db.query(StravaActivity.strava_id)
        .filter(StravaActivity.user_id == user.id)
        .all()
    )

    synced = 0
    for act in activities:
        mapped = _activity_to_dict(act)
        sid = mapped["strava_id"]
        if not sid or sid in existing_ids:
            continue

        # Parse start_date
        raw_start = mapped.get("start_date")
        parsed_start = None
        if raw_start:
            try:
                parsed_start = datetime.fromisoformat(
                    raw_start.replace("Z", "+00:00") if isinstance(raw_start, str) else str(raw_start)
                )
            except (ValueError, TypeError):
                pass

        if not parsed_start:
            continue

        activity = StravaActivity(
            user_id=user.id,
            strava_id=sid,
            name=(mapped.get("name") or "Untitled")[:255],
            activity_type=mapped.get("activity_type", "Ride"),
            start_date=parsed_start,
            timezone=mapped.get("timezone"),
            elapsed_time=mapped.get("elapsed_time"),
            moving_time=mapped.get("moving_time"),
            distance=mapped.get("distance"),
            total_elevation_gain=mapped.get("total_elevation_gain"),
            average_watts=mapped.get("average_watts"),
            max_watts=mapped.get("max_watts"),
            weighted_average_watts=mapped.get("weighted_average_watts"),
            average_heartrate=mapped.get("average_heartrate"),
            max_heartrate=mapped.get("max_heartrate"),
            average_cadence=mapped.get("average_cadence"),
            kilojoules=mapped.get("kilojoules"),
            training_stress_score=mapped.get("training_stress_score"),
            intensity_factor=mapped.get("intensity_factor"),
            training_load=mapped.get("training_load"),
        )
        db.add(activity)
        existing_ids.add(sid)
        synced += 1

    result["synced_activities"] = synced
    print(f"✅ Synced {synced} new activities", file=sys.stderr)

    # ── Step 3: Sync training metrics (CTL/ATL/TSB) ──
    try:
        metrics = await client.get_training_metrics(days=days_back)
    except IntervalsError as e:
        print(f"⚠️ Training metrics sync skipped: {e}", file=sys.stderr)
        return result

    synced_metrics = 0
    existing_metric_dates = set(
        row[0]
        for row in db.query(TrainingMetrics.date)
        .filter(TrainingMetrics.user_id == user.id)
        .all()
    )

    for tm in metrics:
        mapped = _training_metrics_to_dict(tm)
        d_str = mapped.get("date")
        if not d_str:
            continue
        try:
            d = date.fromisoformat(str(d_str)[:10])
        except (ValueError, TypeError):
            continue

        if d in existing_metric_dates:
            continue

        metric = TrainingMetrics(
            user_id=user.id,
            date=d,
            ctl=mapped.get("ctl", 0),
            atl=mapped.get("atl", 0),
            tsb=mapped.get("tsb", 0),
            total_tss=mapped.get("total_tss", 0),
            total_duration_minutes=mapped.get("total_duration_minutes", 0),
            total_distance_km=mapped.get("total_distance_km", 0),
            total_kj=mapped.get("total_kj", 0),
            ride_count=mapped.get("ride_count", 0),
        )
        db.add(metric)
        existing_metric_dates.add(d)
        synced_metrics += 1

    result["synced_metrics"] = synced_metrics
    print(f"✅ Synced {synced_metrics} training metric days", file=sys.stderr)

    return result


# ── CLI ──


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Sync a Discord user's training data from Intervals.icu into the local database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python -m app.sync_intervals joey0624          # Sync last 90 days
  python -m app.sync_intervals joey0624 --days 14  # Sync last 14 days
  python -m app.sync_intervals --list-users        # List registered users
        """,
    )

    parser.add_argument("discord_id", nargs="?", help="Discord username to sync")
    parser.add_argument("--days", type=int, default=90, help="Days of history to sync")
    parser.add_argument("--list-users", action="store_true", help="List registered users")

    args = parser.parse_args()

    if args.list_users:
        from app.user_manager import list_discord_users
        users = list_discord_users()
        if not users:
            print("No Discord users registered.")
            return
        for u in users:
            print(f"  {u['discord_user_id']:20s} {u.get('name', '?'):20s} FTP: {u['ftp']}")
        return

    if not args.discord_id:
        parser.print_help()
        sys.exit(1)

    try:
        result = sync_user(args.discord_id, days_back=args.days)
        print(f"\nSync complete for {args.discord_id}:", file=sys.stderr)
        print(f"  Activities synced:  {result.get('synced_activities', 0)}", file=sys.stderr)
        print(f"  Metric days synced: {result.get('synced_metrics', 0)}", file=sys.stderr)
        if result.get("profile_updates"):
            print(f"  Profile updated:    {result['profile_updates']}", file=sys.stderr)
    except (ValueError, RuntimeError) as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
