"""
Integration tests for multi-user Discord support.

Tests the full lifecycle:
  - app/user_manager.py  — register, lookup, list users by Discord ID
  - app/query_for_user.py — per-user recommendation formatting and context generation

Uses the same SQLite test database and fixtures as test_integration.py.
"""

import pytest
import json
from datetime import date, timedelta
from unittest.mock import patch, MagicMock
from typing import Any, Dict

from app.models import User, TrainingGoal
from app.user_manager import (
    register_discord_user,
    get_user_by_discord,
    list_discord_users,
    get_training_data_for_user,
)
from app.query_for_user import (
    _format_recommendation,
    _build_coaching_context,
)
from app.services.encryption import encrypt


# ── Sample data fixtures ──


@pytest.fixture
def sample_training_data() -> Dict[str, Any]:
    """Simulated data structure as returned by TrainingDataFetcher.fetch_all()."""
    return {
        "athlete": {
            "name": "JohanM",
            "ftp": 284,
            "weight_kg": 78.4,
            "estimated_ftp": None,
        },
        "activities": [
            {
                "name": "Stockholm Road Cycling",
                "start_date": "2026-06-12",
                "tss": 99,
                "distance_km": 70.77,
                "moving_time_seconds": 10020,
                "weighted_avg_watts": 220,
                "classification": {"workout_type_label": "Endurance"},
            },
            {
                "name": "Stockholm - Threshold",
                "start_date": "2026-06-11",
                "tss": 57,
                "distance_km": 8.23,
                "moving_time_seconds": 2340,
                "weighted_avg_watts": 270,
                "classification": {"workout_type_label": "Threshold"},
            },
            {
                "name": "Easy Recovery Spin",
                "start_date": "2026-06-09",
                "tss": 25,
                "distance_km": 15.0,
                "moving_time_seconds": 3600,
                "weighted_avg_watts": 140,
                "classification": {"workout_type_label": "Recovery"},
            },
        ],
        "pmc": [
            {"fitness_ctl": 41.8, "fatigue_atl": 34.0, "form_tsb": 7.7},
            {"fitness_ctl": 43.8, "fatigue_atl": 40.0, "form_tsb": -1.5},
        ],
        "weekly_summary": {
            "rides": 3,
            "total_tss": 181,
            "distance_km": 94.0,
        },
        "fetched_at": "2026-06-16T09:53:53",
    }


# ── Tests for user_manager ──


class TestRegisterDiscordUser:
    """Tests for registering Discord users with encrypted Intervals.icu keys."""

    def test_create_new_user(self, db_session):
        result = register_discord_user(
            discord_user_id="testuser_1",
            intervals_api_key="test-intervals-key-12345",
            athlete_id="Athlete123",
            name="Test Rider",
            ftp=280,
            weight_kg=75.0,
        )
        assert result["action"] == "created"
        assert result["discord_user_id"] == "testuser_1"
        assert result["name"] == "Test Rider"
        assert result["athlete_id"] == "Athlete123"
        assert result["ftp"] == 280
        assert result["weight_kg"] == 75.0
        assert result["has_api_key"] is True
        assert result["id"] > 0

        # Verify it persisted in the DB
        user = db_session.query(User).filter(
            User.discord_user_id == "testuser_1"
        ).first()
        assert user is not None
        assert user.name == "Test Rider"
        assert user.ftp == 280
        # API key should be encrypted (not plaintext)
        assert user.intervals_api_key_encrypted is not None
        assert user.intervals_api_key_encrypted != "test-intervals-key-12345"

    def test_update_existing_user(self, db_session):
        """Registering the same Discord ID should update (upsert)."""
        # Create
        result1 = register_discord_user(
            discord_user_id="testuser_2",
            intervals_api_key="original-key",
            athlete_id="OriginalAthlete",
            name="Original",
            ftp=250,
        )
        user_id = result1["id"]

        # Update with new creds
        result2 = register_discord_user(
            discord_user_id="testuser_2",
            intervals_api_key="updated-key",
            athlete_id="UpdatedAthlete",
            name="Updated",
            ftp=300,
            weight_kg=80.0,
        )
        assert result2["action"] == "updated"
        assert result2["id"] == user_id  # same user
        assert result2["name"] == "Updated"
        assert result2["athlete_id"] == "UpdatedAthlete"
        assert result2["ftp"] == 300
        assert result2["weight_kg"] == 80.0

        # Verify in DB
        user = db_session.query(User).filter(
            User.discord_user_id == "testuser_2"
        ).first()
        assert user.name == "Updated"
        assert user.ftp == 300

    def test_create_uses_defaults(self, db_session):
        """Minimal registration should use sensible defaults."""
        result = register_discord_user(
            discord_user_id="minimal_user",
            intervals_api_key="key123",
            athlete_id="A1",
        )
        assert result["action"] == "created"
        assert result["ftp"] == 200  # default
        assert result["weight_kg"] == 75.0  # default
        assert result["name"] == "minimal_user"  # falls back to discord_id

    def test_upsert_does_not_overwrite_omitted_fields(self, db_session):
        """Omitting optional fields on update should preserve existing values."""
        register_discord_user(
            discord_user_id="partial_update",
            intervals_api_key="key",
            athlete_id="A2",
            name="Full Name",
            ftp=280,
            weight_kg=72.0,
        )
        # Update with only ftp change
        result = register_discord_user(
            discord_user_id="partial_update",
            intervals_api_key="key",  # same key
            athlete_id="A2",
            ftp=290,  # updated
        )
        assert result["ftp"] == 290
        assert result["weight_kg"] == 72.0  # preserved
        assert result["name"] == "Full Name"  # preserved


class TestGetUserByDiscord:
    """Tests for looking up users by Discord ID."""

    def test_find_existing_user(self, db_session):
        register_discord_user(
            discord_user_id="lookup_user",
            intervals_api_key="secret-key-xyz",
            athlete_id="A99",
            name="Find Me",
        )
        user = get_user_by_discord("lookup_user")
        assert user is not None
        assert user["discord_user_id"] == "lookup_user"
        assert user["name"] == "Find Me"
        assert user["intervals_athlete_id"] == "A99"
        # API key should be decrypted
        assert user["intervals_api_key"] == "secret-key-xyz"

    def test_nonexistent_user_returns_none(self):
        user = get_user_by_discord("nobody_here")
        assert user is None

    def test_user_without_discord_id_is_excluded(self, db_session, test_user):
        """A user with discord_user_id=None should not be found by Discord lookup."""
        user = get_user_by_discord(str(test_user.id))
        assert user is None


class TestListDiscordUsers:
    """Tests for listing all registered Discord users."""

    def test_list_multiple_users(self, db_session):
        register_discord_user(
            discord_user_id="user_a", intervals_api_key="ka", athlete_id="AA"
        )
        register_discord_user(
            discord_user_id="user_b", intervals_api_key="kb", athlete_id="BB"
        )
        users = list_discord_users()
        discord_ids = [u["discord_user_id"] for u in users]
        assert "user_a" in discord_ids
        assert "user_b" in discord_ids

    def test_list_empty(self):
        users = list_discord_users()
        assert users == []

    def test_list_excludes_non_discord_users(self, db_session, test_user):
        """Users without discord_user_id should not appear in the list."""
        users = list_discord_users()
        assert all(u["discord_user_id"] is not None for u in users)


class TestGetTrainingDataForUser:
    """Tests for the full training data fetch pipeline.

    These tests mock the Intervals.icu network calls since we don't
    want real HTTP requests in unit tests. The mock returns a known
    data shape that the pipeline should process correctly.
    """

    def test_raises_for_unregistered_user(self):
        with pytest.raises(ValueError, match="No user registered"):
            get_training_data_for_user("unknown_user")

    def test_raises_when_no_api_key(self, db_session):
        """User with discord_user_id but no encrypted key should raise."""
        user = User(
            discord_user_id="keyless_user",
            name="No Key",
            intervals_athlete_id="A1",
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

        with pytest.raises(ValueError, match="no Intervals.icu credentials"):
            get_training_data_for_user("keyless_user")

    @patch("app.data_fetcher.TrainingDataFetcher")
    def test_fetches_data_for_registered_user(self, mock_fetcher_class, db_session):
        """Should call TrainingDataFetcher with the user's decrypted creds."""
        register_discord_user(
            discord_user_id="real_user",
            intervals_api_key="my-api-key",
            athlete_id="MyAthlete",
        )

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_all.return_value = {
            "athlete": {"name": "MyAthlete", "ftp": 280},
            "activities": [],
            "pmc": [],
        }
        mock_fetcher_class.return_value = mock_fetcher

        data = get_training_data_for_user("real_user")

        # Verify the fetcher was constructed with the right credentials
        mock_fetcher_class.assert_called_once_with(
            api_key="my-api-key",
            athlete_id="MyAthlete",
        )
        mock_fetcher.fetch_all.assert_called_once_with(days_back=42)
        assert data["athlete"]["name"] == "MyAthlete"


# ── Tests for query_for_user: _format_recommendation ──


class TestFormatRecommendation:
    """Tests for the human-readable recommendation formatter."""

    def test_daily_recommendation(self, sample_training_data):
        output = _format_recommendation(sample_training_data, template_type="daily")
        assert "DAILY RECOMMENDATION" in output
        assert "JohanM" in output
        assert "284W" in output
        assert "Tempo or Sweet Spot" in output
        assert "TSS:" in output or "TSS" in output
        assert "READY TO GENERATE" in output

    def test_weekly_template(self, sample_training_data):
        output = _format_recommendation(sample_training_data, template_type="weekly")
        assert "WEEKLY" in output.upper()
        assert "WEEKLY TRAINING SUMMARY" in output
        assert "rides:" in output.lower()
        assert "total_tss" in output.lower() or "TSS" in output

    def test_assessment_template(self, sample_training_data):
        output = _format_recommendation(sample_training_data, template_type="assessment")
        assert "FORM ASSESSMENT" in output
        assert "Fresh legs" in output or "training zone" in output
        assert "based on your current form" in output.lower()

    def test_fatigue_form_reads_correctly(self):
        """When TSB is deeply negative, form reading should warn."""
        data = {
            "athlete": {"name": "Test", "ftp": 250},
            "activities": [],
            "pmc": [
                {"fitness_ctl": 50, "fatigue_atl": 80, "form_tsb": -25.0},
            ],
        }
        output = _format_recommendation(data, template_type="daily")
        assert "Deep fatigue" in output or "Accumulated fatigue" in output

    def test_very_fresh_form_reads_correctly(self):
        """Very high TSB should indicate freshness with a detraining note."""
        data = {
            "athlete": {"name": "Test", "ftp": 250},
            "activities": [],
            "pmc": [
                {"fitness_ctl": 30, "fatigue_atl": 10, "form_tsb": 20.0},
            ],
        }
        output = _format_recommendation(data, template_type="daily")
        assert "Very fresh" in output or "detraining" in output

    def test_empty_activities(self):
        """Should not crash when there are no activities."""
        data = {
            "athlete": {"name": "Test", "ftp": 250},
            "activities": [],
            "pmc": [{"fitness_ctl": 40, "fatigue_atl": 30, "form_tsb": 10}],
        }
        output = _format_recommendation(data, template_type="daily")
        assert "DAILY" in output

    def test_none_tss_in_activities(self, sample_training_data):
        """Activities with None TSS should not crash formatting."""
        data = sample_training_data.copy()
        data["activities"] = [
            {
                "name": "Strava Sync",
                "start_date": "2026-06-14",
                "tss": None,
                "distance_km": 0,
                "moving_time_seconds": 0,
                "classification": {},
            },
        ] + data["activities"]
        output = _format_recommendation(data, template_type="daily")
        assert "DAILY" in output  # Should not crash

    def test_no_pmc_data(self):
        """No PMC should still produce a valid output."""
        data = {
            "athlete": {"name": "Test", "ftp": 250},
            "activities": [],
            "pmc": [],
        }
        output = _format_recommendation(data, template_type="daily")
        assert "DAILY" in output

    def test_weekly_with_acr_ratio(self, sample_training_data):
        """Weekly assessment should include acute/chronic ratio."""
        output = _format_recommendation(sample_training_data, template_type="weekly")
        assert "Acute/Chronic" in output or "ACR" in output or "ratio" in output.lower()


# ── Tests for query_for_user: _build_coaching_context ──


class TestBuildCoachingContext:
    """Tests for the structured JSON context builder (for LLM consumption)."""

    def test_context_has_all_sections(self, sample_training_data):
        ctx = _build_coaching_context(sample_training_data)
        assert "athlete" in ctx
        assert "training_load" in ctx
        assert "weekly_summary" in ctx
        assert "recent_activities" in ctx

    def test_athlete_section(self, sample_training_data):
        ctx = _build_coaching_context(sample_training_data)
        assert ctx["athlete"]["name"] == "JohanM"
        assert ctx["athlete"]["ftp"] == 284
        assert ctx["athlete"]["weight_kg"] == 78.4

    def test_training_load_from_pmc(self, sample_training_data):
        ctx = _build_coaching_context(sample_training_data)
        assert ctx["training_load"]["ctl"] == 43.8  # latest PMC entry
        assert ctx["training_load"]["atl"] == 40.0
        assert ctx["training_load"]["tsb"] == -1.5

    def test_training_load_empty_when_no_pmc(self):
        ctx = _build_coaching_context({
            "athlete": {"name": "Test", "ftp": 250},
            "activities": [],
            "pmc": [],
        })
        assert ctx["training_load"] == {}  # empty, no PMC

    def test_weekly_summary(self, sample_training_data):
        ctx = _build_coaching_context(sample_training_data)
        assert ctx["weekly_summary"]["rides"] == 3
        assert ctx["weekly_summary"]["total_tss"] == 181
        assert ctx["weekly_summary"]["distance_km"] == 94.0

    def test_recent_activities_limited_to_5(self, sample_training_data):
        """Should not return more than 5 recent activities."""
        many_activities = []
        for i in range(10):
            many_activities.append({
                "name": f"Activity {i}",
                "start_date": f"2026-06-{10+i:02d}",
                "tss": 50 + i,
                "classification": {},
            })
        data = sample_training_data.copy()
        data["activities"] = many_activities
        ctx = _build_coaching_context(data)
        assert len(ctx["recent_activities"]) <= 5

    def test_activities_sorted_by_date_desc(self, sample_training_data):
        ctx = _build_coaching_context(sample_training_data)
        dates = [a["date"] for a in ctx["recent_activities"]]
        assert dates == sorted(dates, reverse=True), "Activities should be newest first"

    def test_activity_fields_preserved(self, sample_training_data):
        ctx = _build_coaching_context(sample_training_data)
        first = ctx["recent_activities"][0]
        assert "date" in first
        assert "name" in first
        assert "tss" in first
        assert "duration_min" in first
        assert "distance_km" in first
        assert "np" in first

    def test_context_is_json_serializable(self, sample_training_data):
        """The context dict should be serializable to JSON for LLM consumption."""
        ctx = _build_coaching_context(sample_training_data)
        # Should not raise
        json_str = json.dumps(ctx, indent=2)
        assert isinstance(json_str, str)


# ── Tests for the encryption round-trip ──


class TestEncryptionRoundTrip:
    """Verify that encryption/decryption works correctly end-to-end."""

    def test_encrypt_then_decrypt(self):
        """Key should survive a full encrypt/decrypt cycle."""
        plaintext = "my-super-secret-intervals-api-key"
        encrypted = encrypt(plaintext)
        assert encrypted is not None
        assert encrypted != plaintext

        from app.services.encryption import decrypt
        decrypted = decrypt(encrypted)
        assert decrypted == plaintext

    def test_encrypt_produces_different_output_each_time(self):
        """Fernet is non-deterministic — each encryption should differ."""
        plaintext = "same-key"
        encrypted1 = encrypt(plaintext)
        encrypted2 = encrypt(plaintext)
        assert encrypted1 != encrypted2
        # Both should decrypt to the same plaintext
        from app.services.encryption import decrypt
        assert decrypt(encrypted1) == decrypt(encrypted2) == plaintext


# ── Tests for the full CLI pipeline (functions only, no subprocess) ──


class TestFullPipeline:
    """Tests that glue all the pieces together end-to-end (functions, not subprocess).

    These tests mock the Intervals.icu network layer but verify the
    full orchestration: register → fetch → format → context.
    """

    @patch("app.data_fetcher.TrainingDataFetcher")
    def test_register_fetch_format_cycle(
        self,
        mock_fetcher_class,
        db_session,
        sample_training_data,
    ):
        """Full cycle: register a Discord user → fetch their data → format output."""
        # 1. Register
        register_discord_user(
            discord_user_id="e2e_user",
            intervals_api_key="e2e-api-key",
            athlete_id="E2EAthlete",
            name="E2E Rider",
            ftp=275,
        )

        # 2. Mock the fetcher
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_all.return_value = sample_training_data
        mock_fetcher_class.return_value = mock_fetcher

        # 3. Fetch
        data = get_training_data_for_user("e2e_user")
        assert data["athlete"]["name"] == "JohanM"

        # 4. Format
        rec = _format_recommendation(data, template_type="daily")
        assert "DAILY" in rec
        assert "JohanM" in rec

        # 5. Build context
        ctx = _build_coaching_context(data)
        assert ctx["athlete"]["name"] == "JohanM"
        assert ctx["training_load"]["ctl"] == 43.8  # latest PMC entry
        assert ctx["training_load"]["tsb"] == -1.5

    def test_multiple_users_isolated(self, db_session):
        """Two Discord users should have separate registrations."""
        register_discord_user(
            discord_user_id="user_one",
            intervals_api_key="key-1",
            athlete_id="A1",
            name="First",
        )
        register_discord_user(
            discord_user_id="user_two",
            intervals_api_key="key-2",
            athlete_id="A2",
            name="Second",
        )

        u1 = get_user_by_discord("user_one")
        u2 = get_user_by_discord("user_two")

        assert u1["intervals_api_key"] == "key-1"
        assert u2["intervals_api_key"] == "key-2"
        assert u1["intervals_athlete_id"] == "A1"
        assert u2["intervals_athlete_id"] == "A2"
