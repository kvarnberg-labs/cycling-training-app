"""
User Manager — register and look up users by Discord identity.

Handles the full lifecycle:
  - Register a new Discord user + encrypted Intervals.icu credentials
  - Look up a user by Discord user ID
  - Fetch training data on behalf of a specific user

The Discord user ID (the platform handle, e.g. "joey0624") is the primary
lookup key. Each Discord user maps to exactly one Intervals.icu profile.

Usage (CLI):
    python -m app.user_manager register joey0624 --intervals-key "xxx" --athlete-id "JohanM"
    python -m app.user_manager list
    python -m app.user_manager lookup joey0624

Usage (API):
    from app.user_manager import get_training_data_for_user
    data = get_training_data_for_user("joey0624", days_back=42)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.database import SessionLocal, get_db
from app.models import User
from app.services.encryption import encrypt, decrypt

logger = logging.getLogger(__name__)

# Silence noisy SQLAlchemy INFO logs in CLI mode
if not logger.handlers:
    pass  # root logger will be configured in main()


# ── Database helpers ──


def _get_session() -> Session:
    """Get a new database session."""
    return SessionLocal()


# ── Public API ──


def register_discord_user(
    discord_user_id: str,
    intervals_api_key: str,
    athlete_id: str,
    name: Optional[str] = None,
    ftp: Optional[int] = None,
    weight_kg: Optional[float] = None,
) -> Dict[str, Any]:
    """Register or update a Discord user with Intervals.icu credentials.

    The API key is encrypted at rest using Fernet (via app's SECRET_KEY).
    If a user with this Discord ID already exists, their credentials are
    updated (upsert).

    Args:
        discord_user_id: Discord username/handle (e.g. "joey0624")
        intervals_api_key: Plaintext Intervals.icu API key
        athlete_id: Intervals.icu athlete ID (e.g. "JohanM")
        name: Optional display name
        ftp: Optional FTP value (watts)
        weight_kg: Optional weight (kg)

    Returns:
        Dict with user info (sans plaintext API key)
    """
    encrypted_key = encrypt(intervals_api_key)
    if not encrypted_key:
        raise RuntimeError(
            "Failed to encrypt API key. Check SECRET_KEY is configured."
        )

    db = _get_session()
    try:
        user = db.query(User).filter(
            User.discord_user_id == discord_user_id
        ).first()

        if user:
            # Update existing
            user.intervals_api_key_encrypted = encrypted_key
            user.intervals_athlete_id = athlete_id
            if name:
                user.name = name
            if ftp is not None:
                user.ftp = ftp
            if weight_kg is not None:
                user.weight_kg = weight_kg
            action = "updated"
        else:
            # Create new
            user = User(
                discord_user_id=discord_user_id,
                name=name or discord_user_id,
                intervals_api_key_encrypted=encrypted_key,
                intervals_athlete_id=athlete_id,
                ftp=ftp or 200,
                weight_kg=weight_kg or 75.0,
            )
            db.add(user)
            action = "created"

        db.commit()
        db.refresh(user)

        return {
            "id": user.id,
            "discord_user_id": user.discord_user_id,
            "name": user.name,
            "athlete_id": user.intervals_athlete_id,
            "has_api_key": bool(user.intervals_api_key_encrypted),
            "ftp": user.ftp,
            "weight_kg": user.weight_kg,
            "action": action,
        }
    finally:
        db.close()


def get_user_by_discord(
    discord_user_id: str,
) -> Optional[Dict[str, Any]]:
    """Look up a user by Discord ID.

    Returns user info dict with the API key DECRYPTED (use with care).
    Returns None if no user is registered with that Discord ID.

    The returned dict includes:
      - id, discord_user_id, name
      - intervals_api_key (decrypted, for API calls)
      - intervals_athlete_id
      - ftp, weight_kg
    """
    db = _get_session()
    try:
        user = db.query(User).filter(
            User.discord_user_id == discord_user_id
        ).first()

        if not user:
            return None

        decrypted_key = None
        if user.intervals_api_key_encrypted:
            decrypted_key = decrypt(user.intervals_api_key_encrypted)

        return {
            "id": user.id,
            "discord_user_id": user.discord_user_id,
            "name": user.name,
            "intervals_api_key": decrypted_key,
            "intervals_athlete_id": user.intervals_athlete_id,
            "ftp": user.ftp,
            "weight_kg": user.weight_kg,
        }
    finally:
        db.close()


def list_discord_users() -> list[Dict[str, Any]]:
    """List all registered Discord users (without API keys)."""
    db = _get_session()
    try:
        users = db.query(User).filter(
            User.discord_user_id.isnot(None)
        ).all()
        return [
            {
                "id": u.id,
                "discord_user_id": u.discord_user_id,
                "name": u.name,
                "athlete_id": u.intervals_athlete_id,
                "has_api_key": bool(u.intervals_api_key_encrypted),
                "ftp": u.ftp,
                "weight_kg": u.weight_kg,
            }
            for u in users
        ]
    finally:
        db.close()


def get_training_data_for_user(
    discord_user_id: str,
    days_back: int = 42,
) -> Dict[str, Any]:
    """Fetch Intervals.icu training data for a specific Discord user.

    Looks up the user's credentials, fetches their training data,
    and returns the full data dict (athlete profile + activities + PMC).

    Args:
        discord_user_id: Discord username/handle
        days_back: Days of training history to fetch

    Returns:
        Full training data dict from ``TrainingDataFetcher.fetch_all()``
        with the athlete info also included for direct access.

    Raises:
        ValueError: If user not found or no API key configured
        RuntimeError: If Intervals.icu fetch fails
    """
    user_info = get_user_by_discord(discord_user_id)
    if not user_info:
        raise ValueError(
            f"No user registered with Discord ID '{discord_user_id}'. "
            "Register them first with: "
            f"python -m app.user_manager register {discord_user_id} ..."
        )

    api_key = user_info.get("intervals_api_key")
    athlete_id = user_info.get("intervals_athlete_id")

    if not api_key or not athlete_id:
        raise ValueError(
            f"User '{discord_user_id}' has no Intervals.icu credentials configured."
        )

    # Fetch via data_fetcher, passing per-user credentials
    from app.data_fetcher import TrainingDataFetcher

    fetcher = TrainingDataFetcher(
        api_key=api_key,
        athlete_id=athlete_id,
    )
    data = fetcher.fetch_all(days_back=days_back)

    # Ensure athlete info is top-level (data_fetcher puts it under "athlete")
    return data


# ── CLI ──


def _cli():
    """CLI entrypoint for user management."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Manage Discord ↔ Intervals.icu user registrations",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Register
    reg = sub.add_parser("register", help="Register or update a Discord user")
    reg.add_argument("discord_id", help="Discord username (e.g. joey0624)")
    reg.add_argument("--intervals-key", required=True, help="Intervals.icu API key")
    reg.add_argument("--athlete-id", required=True, help="Intervals.icu athlete ID")
    reg.add_argument("--name", help="Display name")
    reg.add_argument("--ftp", type=int, help="FTP in watts")
    reg.add_argument("--weight", type=float, help="Weight in kg")

    # Lookup
    lookup = sub.add_parser("lookup", help="Look up a Discord user")
    lookup.add_argument("discord_id", help="Discord username")
    lookup.add_argument("--show-key", action="store_true",
                        help="Show decrypted API key (dangerous!)")

    # List
    sub.add_parser("list", help="List all registered Discord users")

    # Fetch
    fetch = sub.add_parser("fetch", help="Fetch training data for a Discord user")
    fetch.add_argument("discord_id", help="Discord username")
    fetch.add_argument("--days", type=int, default=14,
                       help="Days of training history (default: 14)")
    fetch.add_argument("--summary", action="store_true",
                       help="Show only summary stats")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    if args.command == "register":
        result = register_discord_user(
            discord_user_id=args.discord_id,
            intervals_api_key=args.intervals_key,
            athlete_id=args.athlete_id,
            name=args.name,
            ftp=args.ftp,
            weight_kg=args.weight,
        )
        print(f"✅ User {result['action']}: {result['name']} (Discord: {result['discord_user_id']})")
        print(f"   Athlete ID: {result['athlete_id']}")
        print(f"   FTP: {result['ftp']}W, Weight: {result['weight_kg']}kg")

    elif args.command == "lookup":
        user = get_user_by_discord(args.discord_id)
        if not user:
            print(f"❌ No user found with Discord ID '{args.discord_id}'")
            sys.exit(1)
        print(f"User: {user['name']} (Discord: {user['discord_user_id']})")
        print(f"Athlete ID: {user['intervals_athlete_id']}")
        print(f"FTP: {user['ftp']}W, Weight: {user['weight_kg']}kg")
        print(f"Has API key: {bool(user['intervals_api_key'])}")
        if args.show_key and user.get("intervals_api_key"):
            print(f"API Key (decrypted): {user['intervals_api_key']}")

    elif args.command == "list":
        users = list_discord_users()
        if not users:
            print("No Discord users registered.")
            return
        print(f"{'#':>3} {'Discord ID':20s} {'Name':20s} {'Athlete ID':15s} {'FTP':>5} {'Key':>5}")
        print("-" * 70)
        for u in users:
            print(f"{u['id']:3d} {u['discord_user_id']:20s} {(u['name'] or ''):20s} "
                  f"{(u['athlete_id'] or ''):15s} {u['ftp']:5d} {'✓' if u['has_api_key'] else '✗':>5}")

    elif args.command == "fetch":
        try:
            data = get_training_data_for_user(args.discord_id, days_back=args.days)
        except (ValueError, RuntimeError) as e:
            print(f"❌ {e}", file=sys.stderr)
            sys.exit(1)

        athlete = data.get("athlete", {})
        activities = data.get("activities", [])

        print(f"✅ Data for {athlete.get('name', '?')} (FTP: {athlete.get('ftp', '?')}W)")
        print(f"   {len(activities)} activities in {args.days} days")

        if args.summary:
            pmc = data.get("pmc", [])
            if pmc:
                latest = pmc[-1]
                print(f"   CTL: {latest.get('fitness_ctl', '?'):.1f}")
                print(f"   ATL: {latest.get('fatigue_atl', '?'):.1f}")
                print(f"   TSB: {latest.get('form_tsb', '?'):.1f}")
            week_tss = sum(a.get("tss", 0) or 0 for a in activities)
            week_dist = sum(a.get("distance_km", 0) or 0 for a in activities)
            print(f"   Total TSS: {week_tss:.0f}")
            print(f"   Total distance: {week_dist:.1f} km")
        else:
            print(json.dumps(data, indent=2, default=str))


if __name__ == "__main__":
    _cli()
