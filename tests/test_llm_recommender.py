"""Tests for the LLM-based workout recommender service using the OpenAI Agents SDK.

Tests the context builder, agent initialisation, and output model validation.
The actual LLM call testing is handled by the SDK — we test the integration layer.
"""

import json
from datetime import date, datetime
from typing import Any, Dict, List

import pytest

from app.services.llm_recommender import (
    _build_athlete_context,
    _summarize_weekly_training,
    WeeklyWorkoutPlan,
    WorkoutRecommendation,
    generate_llm_plan,
)
from app.config import settings


class TestBuildAthleteContext:
    """Tests for building athlete context for the agent input."""

    def test_basic_context_no_data(self):
        """Context should still work with minimal data."""
        context = _build_athlete_context(
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
        context = _build_athlete_context(
            user_profile={"ftp": 200, "weight_kg": 75, "training_goal": "build"},
            training_metrics={"ctl": 60, "atl": 45, "tsb": 15},
            recent_activities=[],
            existing_scheduled=[],
        )
        assert "CTL (Fitness): 60.0" in context
        assert "ATL (Fatigue): 45.0" in context
        assert "TSB (Form): 15.0" in context
        # TSB 15 falls in the "peaking" range (15 <= tsb)
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
        context = _build_athlete_context(
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
        context = _build_athlete_context(
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
        context = _build_athlete_context(
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
        context = _build_athlete_context(
            user_profile={"ftp": 200, "weight_kg": 75, "training_goal": "race"},
            training_metrics={"ctl": 80, "atl": 110, "tsb": -30},
            recent_activities=[],
            existing_scheduled=[],
        )
        assert "Deep fatigue zone" in context

    def test_peaking_interpretation(self):
        """Should flag peaking for very positive TSB."""
        context = _build_athlete_context(
            user_profile={"ftp": 200, "weight_kg": 75, "training_goal": "race"},
            training_metrics={"ctl": 80, "atl": 50, "tsb": 30},
            recent_activities=[],
            existing_scheduled=[],
        )
        assert "Peaking" in context

    def test_profile_with_location(self):
        """Should include location when lat/lon are set."""
        context = _build_athlete_context(
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


class TestWeeklyTrainingSummary:
    """Tests for the weekly training summary helper."""

    def test_summarises_by_iso_week(self):
        """Should group activities by ISO week."""
        activities = [
            {
                "start_date": "2026-06-01T10:00:00Z",
                "moving_time": 7200,
                "distance": 40000,
                "training_stress_score": 100,
            },
            {
                "start_date": "2026-06-03T10:00:00Z",
                "moving_time": 5400,
                "distance": 30000,
                "training_stress_score": 80,
            },
        ]
        summary = _summarize_weekly_training(activities)
        assert len(summary) >= 1
        # Both activities should be in the same week
        week_entry = [s for s in summary if "2026-W23" in s["week_label"]]
        assert len(week_entry) == 1
        assert week_entry[0]["ride_count"] == 2
        assert week_entry[0]["total_tss"] == 180.0
        # 7200 + 5400 = 12600 seconds / 3600 = 3.5h
        assert week_entry[0]["total_hours"] == 3.5
        # 40km + 30km = 70km
        assert week_entry[0]["total_distance_km"] == 70.0

    def test_empty_activities(self):
        """Empty list should return empty summary."""
        assert _summarize_weekly_training([]) == []

    def test_missing_start_date_skipped(self):
        """Activities without start_date should be skipped."""
        activities = [{"moving_time": 3600, "distance": 20000}]
        assert _summarize_weekly_training(activities) == []


class TestWorkoutRecommendationModel:
    """Tests for the Pydantic output model used by the agent."""

    def test_valid_recommendation(self):
        """A complete, valid recommendation should pass Pydantic validation."""
        w = WorkoutRecommendation(
            scheduled_date="2026-06-02",
            workout_type="endurance",
            title="Long Ride",
            description="Steady Zone 2 effort",
            duration_minutes=120,
            target_power_zone="Zone 2 (56-75% FTP)",
            target_rpe=3,
            is_indoor=False,
        )
        assert w.scheduled_date == "2026-06-02"
        assert w.workout_type == "endurance"
        assert w.duration_minutes == 120

    def test_default_values(self):
        """Should use sensible defaults for optional fields."""
        w = WorkoutRecommendation(
            scheduled_date="2026-06-02",
            workout_type="recovery",
            title="Recovery Spin",
        )
        assert w.description == ""
        assert w.duration_minutes == 60
        assert w.target_power_zone == ""
        assert w.target_rpe is None
        assert w.is_indoor is False

    def test_valid_workout_types(self):
        """All valid workout types should be accepted."""
        valid_types = [
            "recovery", "endurance", "tempo", "threshold",
            "vo2max", "sprint", "interval",
        ]
        for wtype in valid_types:
            w = WorkoutRecommendation(
                scheduled_date="2026-06-02",
                workout_type=wtype,
                title="Test",
            )
            assert w.workout_type == wtype

    def test_duration_bounds_enforced(self):
        """Duration should be clamped by Pydantic validators (ge=20, le=300)."""
        with pytest.raises(ValueError):
            WorkoutRecommendation(
                scheduled_date="2026-06-02",
                workout_type="endurance",
                title="Too Long",
                duration_minutes=999,
            )
        with pytest.raises(ValueError):
            WorkoutRecommendation(
                scheduled_date="2026-06-02",
                workout_type="endurance",
                title="Too Short",
                duration_minutes=5,
            )

    def test_rpe_bounds(self):
        """RPE should be 1-10."""
        with pytest.raises(ValueError):
            WorkoutRecommendation(
                scheduled_date="2026-06-02",
                workout_type="endurance",
                title="Bad RPE",
                target_rpe=15,
            )


class TestWeeklyWorkoutPlanModel:
    """Tests for the weekly plan container model."""

    def test_valid_plan(self):
        """A plan with valid workouts should pass."""
        plan = WeeklyWorkoutPlan(workouts=[
            WorkoutRecommendation(
                scheduled_date="2026-06-02",
                workout_type="endurance",
                title="Endurance Ride",
            ),
            WorkoutRecommendation(
                scheduled_date="2026-06-03",
                workout_type="threshold",
                title="Threshold Intervals",
            ),
        ])
        assert len(plan.workouts) == 2

    def test_empty_plan_fails(self):
        """A plan with no workouts should fail validation."""
        with pytest.raises(ValueError):
            WeeklyWorkoutPlan(workouts=[])

    def test_too_many_workouts_fails(self):
        """A plan with more than 10 workouts should fail."""
        with pytest.raises(ValueError):
            WeeklyWorkoutPlan(workouts=[
                WorkoutRecommendation(
                    scheduled_date="2026-06-02",
                    workout_type="endurance",
                    title=f"Workout {i}",
                ) for i in range(15)
            ])


class TestGenerateLLMPlan:
    """Tests for the main entry point that calls the agent."""

    def test_returns_none_when_not_configured(self):
        """Should return None when LLM is not configured (no API key/base)."""
        # Temporarily clear the settings
        original_key = settings.llm_api_key
        original_base = settings.llm_api_base
        settings.llm_api_key = ""
        settings.llm_api_base = ""

        # Reset the lazy-initialised agent
        import app.services.llm_recommender as mod
        mod._client_initialised = False
        mod._agent = None

        result = None
        import asyncio
        try:
            result = asyncio.run(generate_llm_plan(
                user_id=1,
                user_profile={"ftp": 200, "weight_kg": 75, "training_goal": "base"},
                training_metrics={"ctl": 50, "atl": 40, "tsb": 10},
                recent_activities=[],
                existing_scheduled=[],
                week_start=date(2026, 6, 1),
            ))
        finally:
            # Restore
            settings.llm_api_key = original_key
            settings.llm_api_base = original_base
            mod._client_initialised = False
            mod._agent = None

        assert result is None

    def test_creates_proper_input_format(self, monkeypatch):
        """When called with LLM configured, should pass proper input to the agent."""
        api_key = settings.llm_api_key
        api_base = settings.llm_api_base
        if not api_key or not api_base:
            pytest.skip("No LLM configured in .env — skipping integration test")

        # Reset lazy init so the agent gets created with real config
        import app.services.llm_recommender as mod
        mod._client_initialised = False
        mod._agent = None

        import asyncio
        result = asyncio.run(generate_llm_plan(
            user_id=1,
            user_profile={
                "ftp": 240,
                "weight_kg": 72,
                "training_goal": "build",
                "resting_hr": 55,
                "max_hr": 188,
            },
            training_metrics={"ctl": 65, "atl": 50, "tsb": 15},
            recent_activities=[
                {
                    "start_date": "2026-06-01T10:00:00Z",
                    "activity_type": "Ride",
                    "name": "Weekend Group Ride",
                    "moving_time": 10800,
                    "distance": 80000,
                    "average_watts": 195,
                    "weighted_average_watts": 205,
                    "average_heartrate": 142,
                    "total_elevation_gain": 450,
                    "training_stress_score": 150.0,
                    "intensity_factor": 0.85,
                },
                {
                    "start_date": "2026-06-03T07:00:00Z",
                    "activity_type": "VirtualRide",
                    "name": "Zwift Race",
                    "moving_time": 5400,
                    "distance": 35000,
                    "average_watts": 220,
                    "weighted_average_watts": 235,
                    "average_heartrate": 158,
                    "total_elevation_gain": 200,
                    "training_stress_score": 110.0,
                    "intensity_factor": 0.98,
                },
            ],
            existing_scheduled=[],
            week_start=date(2026, 6, 8),
            weather_forecasts={
                "2026-06-08": {
                    "label": "Rain",
                    "symbol": "rain",
                    "temp_min": 10.0,
                    "temp_max": 15.0,
                    "precipitation_mm": 8.0,
                    "wind_speed_ms": 6.0,
                    "indoor": True,
                    "outdoor": False,
                },
                "2026-06-09": {
                    "label": "Clear",
                    "symbol": "clear",
                    "temp_min": 12.0,
                    "temp_max": 20.0,
                    "precipitation_mm": 0.0,
                    "wind_speed_ms": 3.0,
                    "indoor": False,
                    "outdoor": True,
                },
            },
        ))

        assert result is not None, "Agent should return a plan when LLM is configured"
        assert len(result) > 0, "Should contain at least one workout"
        assert len(result) <= 10, "Should not exceed 10 workouts"

        # Verify the format matches what the endpoint expects
        for w in result:
            assert "user_id" in w
            assert "scheduled_date" in w
            assert "workout_type" in w
            assert "title" in w
            assert w["workout_type"] in {
                "recovery", "endurance", "tempo", "threshold",
                "vo2max", "sprint", "interval",
            }
            assert isinstance(w["is_indoor"], bool)
            assert w["status"] == "suggested"
            assert w["source"] == "recommendation"
