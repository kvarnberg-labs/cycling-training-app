"""Tests for the LLM-based workout recommender service."""

import json
from datetime import date, datetime
from typing import Any, Dict, List

import pytest

from app.services.llm_recommender import (
    build_athlete_context,
    build_recommendation_prompt,
    parse_llm_response,
    validate_workout,
)


class TestBuildAthleteContext:
    """Tests for building athlete context for the LLM prompt."""

    def test_basic_context_no_data(self):
        """Context should still work with minimal data."""
        context = build_athlete_context(
            user_profile={"ftp": 250, "weight_kg": 72, "training_goal": "base"},
            training_metrics=None,
            recent_activities=[],
            existing_scheduled=[],
        )
        assert "ATHLETE PROFILE" in context
        assert "250" in context
        assert "base" in context
        assert "RECENT STRAVA ACTIVITIES" in context
        assert "No recent Strava activities synced" in context

    def test_context_with_training_metrics(self):
        """Should include CTL/ATL/TSB and interpretation."""
        context = build_athlete_context(
            user_profile={"ftp": 200, "weight_kg": 75, "training_goal": "build"},
            training_metrics={"ctl": 60, "atl": 45, "tsb": 15},
            recent_activities=[],
            existing_scheduled=[],
        )
        assert "CTL (Fitness): 60.0" in context
        assert "ATL (Fatigue): 45.0" in context
        assert "TSB (Form): 15.0" in context
        # TSB 15 is in the "peaking" range (15 <= tsb)
        assert "Peaking" in context

    def test_context_with_strava_activities(self):
        """Should format recent Strava activities as a table."""
        activities = [
            {
                "start_date": "2026-05-28T10:00:00Z",
                "activity_type": "Ride",
                "name": "Morning Ride",
                "moving_time": 7200,
                "distance": 40000,
                "average_watts": 200,
                "weighted_average_watts": 210,
                "average_heartrate": 145,
                "total_elevation_gain": 350,
                "training_stress_score": 120.0,
                "intensity_factor": 0.85,
            }
        ]
        context = build_athlete_context(
            user_profile={"ftp": 250, "weight_kg": 75, "training_goal": "build"},
            training_metrics={"ctl": 50, "atl": 40, "tsb": 10},
            recent_activities=activities,
            existing_scheduled=[],
        )
        assert "2026-05-28" in context
        assert "Morning Ride" in context
        assert "Ride" in context
        assert "40.0km" in context  # 40000m -> 40km
        assert "120min" in context  # 7200s -> 120min

    def test_context_with_existing_scheduled(self):
        """Should list already-scheduled workouts."""
        existing = [
            {
                "scheduled_date": date(2026, 6, 2),
                "title": "Tempo Ride",
                "workout_type": "tempo",
                "duration_minutes": 90,
            }
        ]
        context = build_athlete_context(
            user_profile={"ftp": 200, "weight_kg": 75, "training_goal": "base"},
            training_metrics=None,
            recent_activities=[],
            existing_scheduled=existing,
        )
        assert "ALREADY SCHEDULED" in context
        assert "2026-06-02" in context
        assert "Tempo Ride" in context

    def test_context_with_weather(self):
        """Should include weather forecast in context."""
        weather = {
            "2026-06-02": {
                "symbol": "rain",
                "label": "Rain",
                "temp_min": 8.0,
                "temp_max": 14.0,
                "precipitation_mm": 5.0,
                "wind_speed_ms": 7.0,
                "indoor": True,
                "outdoor": False,
            }
        }
        context = build_athlete_context(
            user_profile={"ftp": 200, "weight_kg": 75, "training_goal": "base"},
            training_metrics=None,
            recent_activities=[],
            existing_scheduled=[],
            weather_forecasts=weather,
        )
        assert "WEATHER FORECAST" in context
        assert "Rain" in context
        assert "Indoor" in context
        assert "8.0..14.0°C" in context

    def test_deep_fatigue_interpretation(self):
        """Should flag deep fatigue for very negative TSB."""
        context = build_athlete_context(
            user_profile={"ftp": 200, "weight_kg": 75, "training_goal": "race"},
            training_metrics={"ctl": 80, "atl": 110, "tsb": -30},
            recent_activities=[],
            existing_scheduled=[],
        )
        assert "Deep fatigue zone" in context

    def test_peaking_interpretation(self):
        """Should flag peaking for very positive TSB."""
        context = build_athlete_context(
            user_profile={"ftp": 200, "weight_kg": 75, "training_goal": "race"},
            training_metrics={"ctl": 80, "atl": 50, "tsb": 30},
            recent_activities=[],
            existing_scheduled=[],
        )
        assert "Peaking" in context

    def test_profile_with_location(self):
        """Should include location when lat/lon are set."""
        context = build_athlete_context(
            user_profile={
                "ftp": 200,
                "weight_kg": 75,
                "training_goal": "base",
                "location_lat": 59.33,
                "location_lon": 18.07,
            },
            training_metrics=None,
            recent_activities=[],
            existing_scheduled=[],
        )
        assert "lat 59.33" in context
        assert "lon 18.07" in context


class TestBuildRecommendationPrompt:
    """Tests for the full prompt builder."""

    def test_prompt_has_correct_structure(self):
        """Should return system + user messages."""
        context = build_athlete_context(
            user_profile={"ftp": 200, "weight_kg": 75, "training_goal": "base"},
            training_metrics={"ctl": 50, "atl": 40, "tsb": 10},
            recent_activities=[],
            existing_scheduled=[],
        )
        messages = build_recommendation_prompt(
            week_start=date(2026, 6, 1),
            athlete_context=context,
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "expert cycling coach" in messages[0]["content"].lower()
        assert messages[1]["role"] == "user"
        assert "2026-06-01" in messages[1]["content"]

    def test_output_format_instructions_included(self):
        """Should include JSON output format instructions."""
        context = build_athlete_context(
            user_profile={"ftp": 200, "weight_kg": 75, "training_goal": "base"},
            training_metrics=None,
            recent_activities=[],
            existing_scheduled=[],
        )
        messages = build_recommendation_prompt(
            week_start=date(2026, 6, 1),
            athlete_context=context,
        )
        user_msg = messages[1]["content"]
        assert "scheduled_date" in user_msg
        assert "workout_type" in user_msg
        assert "duration_minutes" in user_msg


class TestParseLLMResponse:
    """Tests for parsing LLM JSON responses."""

    def test_parse_direct_json_array(self):
        """Should parse a bare JSON array."""
        response = json.dumps([
            {
                "scheduled_date": "2026-06-02",
                "workout_type": "endurance",
                "title": "Endurance Ride",
                "description": "Steady Zone 2",
                "duration_minutes": 120,
                "target_power_zone": "Zone 2",
                "target_rpe": 3,
                "is_indoor": False,
            }
        ])
        result = parse_llm_response(response)
        assert result is not None
        assert len(result) == 1
        assert result[0]["workout_type"] == "endurance"

    def test_parse_json_in_code_fence(self):
        """Should extract JSON from markdown code fences."""
        response = """Here is the plan:

```json
[
  {
    "scheduled_date": "2026-06-02",
    "workout_type": "endurance",
    "title": "Easy Endurance",
    "description": "Zone 2 ride",
    "duration_minutes": 90,
    "target_power_zone": "Zone 2",
    "target_rpe": 3,
    "is_indoor": false
  }
]
```"""
        result = parse_llm_response(response)
        assert result is not None
        assert len(result) == 1
        assert result[0]["title"] == "Easy Endurance"

    def test_parse_wrapped_in_workouts_key(self):
        """Should handle JSON with a 'workouts' wrapper key."""
        response = json.dumps({
            "workouts": [
                {
                    "scheduled_date": "2026-06-02",
                    "workout_type": "recovery",
                    "title": "Recovery Spin",
                    "description": "Easy spin",
                    "duration_minutes": 45,
                    "target_power_zone": "Zone 1",
                    "target_rpe": 2,
                    "is_indoor": True,
                }
            ]
        })
        result = parse_llm_response(response)
        assert result is not None
        assert len(result) == 1
        assert result[0]["workout_type"] == "recovery"

    def test_parse_with_recommendations_key(self):
        """Should handle 'recommendations' wrapper key."""
        response = json.dumps({
            "recommendations": [
                {
                    "scheduled_date": "2026-06-03",
                    "workout_type": "threshold",
                    "title": "Threshold Intervals",
                    "description": "3x12min at FTP",
                    "duration_minutes": 75,
                    "target_power_zone": "Threshold",
                    "target_rpe": 7,
                    "is_indoor": False,
                }
            ]
        })
        result = parse_llm_response(response)
        assert result is not None
        assert len(result) == 1

    def test_parse_empty_response_returns_none(self):
        """Empty or None response should return None."""
        assert parse_llm_response(None) is None
        assert parse_llm_response("") is None
        assert parse_llm_response("I couldn't generate a plan") is None

    def test_parse_invalid_json_returns_none(self):
        """Malformed JSON should return None."""
        assert parse_llm_response("{invalid json") is None

    def test_parse_empty_array_returns_empty(self):
        """Empty array should return empty list."""
        result = parse_llm_response("[]")
        assert result == []


class TestValidateWorkout:
    """Tests for workout validation."""

    def test_valid_workout(self):
        """A complete, valid workout should pass."""
        assert validate_workout({
            "scheduled_date": "2026-06-02",
            "workout_type": "endurance",
            "title": "Long Ride",
            "description": "Zone 2",
            "duration_minutes": 120,
        })

    def test_missing_required_fields(self):
        """Missing scheduled_date, workout_type, or title should fail."""
        assert not validate_workout({
            "workout_type": "endurance",
            "title": "Ride",
        })
        assert not validate_workout({
            "scheduled_date": "2026-06-02",
            "title": "Ride",
        })
        assert not validate_workout({
            "scheduled_date": "2026-06-02",
            "workout_type": "endurance",
        })

    def test_invalid_workout_type(self):
        """Unknown workout types should be rejected."""
        assert not validate_workout({
            "scheduled_date": "2026-06-02",
            "workout_type": "yoga",
            "title": "Yoga Session",
        })

    def test_all_valid_types(self):
        """All valid workout types should pass."""
        valid_types = ["recovery", "endurance", "tempo", "threshold", "vo2max", "sprint", "interval"]
        for wtype in valid_types:
            assert validate_workout({
                "scheduled_date": "2026-06-02",
                "workout_type": wtype,
                "title": "Test Workout",
            }), f"Type {wtype} should be valid"
