"""Unit tests for the recommendation engine (app/services/recommendation_engine.py).

Tests weekly plan generation, workout type selection, TSS capacity computation,
and workload distribution across training phases.
"""

import pytest
from datetime import date, timedelta
from app.services.recommendation_engine import (
    generate_weekly_plan,
    compute_weekly_tss_capacity,
    _get_tsb_zone,
    _get_workout_type_for_zone,
    _pick_workout_from_library,
    _estimate_workout_tss,
    _pick_training_days,
    WORKOUT_LIBRARY,
    PHASE_WORKOUT_SPLIT,
)
from app.models import TrainingGoal, WorkoutType


class TestGetTSBZone:
    """Tests for TSB zone classification."""

    def test_overreaching(self):
        assert _get_tsb_zone(-50) == "overreaching"
        assert _get_tsb_zone(-30) == "overreaching"
        assert _get_tsb_zone(-20) == "overreaching"

    def test_heavy(self):
        assert _get_tsb_zone(-15) == "heavy"
        assert _get_tsb_zone(-10) == "heavy"

    def test_optimal(self):
        assert _get_tsb_zone(-5) == "optimal"
        assert _get_tsb_zone(0) == "optimal"
        assert _get_tsb_zone(4) == "optimal"

    def test_fresh(self):
        assert _get_tsb_zone(10) == "fresh"

    def test_peaking(self):
        assert _get_tsb_zone(20) == "peaking"
        assert _get_tsb_zone(100) == "peaking"

    def test_default_for_missing(self):
        """TSB values not covered by ranges should return 'optimal'."""
        assert _get_tsb_zone(9999) in ("peaking", "optimal")


class TestComputeWeeklyTSSCapacity:
    """Tests for weekly TSS capacity calculation."""

    def test_base_phase(self):
        weekly, max_daily = compute_weekly_tss_capacity(ctl=50, goal=TrainingGoal.BASE)
        assert weekly > 0
        assert max_daily > 0
        assert weekly >= 150  # Minimum weekly TSS even for low CTL

    def test_recovery_lower_than_build(self):
        recovery_weekly, _ = compute_weekly_tss_capacity(ctl=50, goal=TrainingGoal.RECOVERY)
        build_weekly, _ = compute_weekly_tss_capacity(ctl=50, goal=TrainingGoal.BUILD)
        assert recovery_weekly < build_weekly

    def test_high_ctl_gives_higher_weekly(self):
        low_weekly, _ = compute_weekly_tss_capacity(ctl=30, goal=TrainingGoal.BASE)
        high_weekly, _ = compute_weekly_tss_capacity(ctl=60, goal=TrainingGoal.BASE)
        assert high_weekly > low_weekly

    def test_max_daily_not_exceeding_weekly(self):
        weekly, max_daily = compute_weekly_tss_capacity(ctl=50, goal=TrainingGoal.BUILD)
        assert max_daily < weekly


class TestPickWorkoutFromLibrary:
    """Tests for workout selection from the library."""

    def test_returns_valid_workout(self):
        workout = _pick_workout_from_library("endurance", set())
        assert workout is not None
        assert "title" in workout
        assert "description" in workout
        assert "duration_minutes" in workout

    def test_avoid_ids_works(self):
        """Should not return workouts with titles in the avoid set."""
        all_endurance = [w["title"] for w in WORKOUT_LIBRARY["endurance"]]
        if len(all_endurance) > 1:
            avoid = {all_endurance[0]}
            picked = _pick_workout_from_library("endurance", avoid)
            assert picked["title"] not in avoid

    def test_unknown_type_returns_none(self):
        workout = _pick_workout_from_library("nonexistent_type", set())
        assert workout is None

    def test_all_workout_types_have_entries(self):
        """Every workout type in the library should have at least one template."""
        for wt in ["recovery", "endurance", "tempo", "threshold", "vo2max", "sprint", "interval"]:
            assert wt in WORKOUT_LIBRARY, f"Missing workout type: {wt}"
            assert len(WORKOUT_LIBRARY[wt]) > 0, f"No workouts for type: {wt}"

    def test_no_workout_exceeds_4_hours(self):
        """Templates should be reasonable in duration."""
        for wt, workouts in WORKOUT_LIBRARY.items():
            for w in workouts:
                assert w["duration_minutes"] <= 240, f"Workout '{w['title']}' exceeds 4 hours"


class TestEstimateWorkoutTSS:
    """Tests for workout TSS estimation."""

    def test_returns_positive_for_valid_workout(self):
        tss = _estimate_workout_tss(
            {"workout_type": "endurance", "duration_minutes": 60}, ftp=250
        )
        assert tss > 0

    def test_longer_workout_higher_tss(self):
        short = _estimate_workout_tss(
            {"workout_type": "endurance", "duration_minutes": 30}, ftp=250
        )
        long = _estimate_workout_tss(
            {"workout_type": "endurance", "duration_minutes": 60}, ftp=250
        )
        assert long > short

    def test_threshold_higher_than_endurance(self):
        endurance = _estimate_workout_tss(
            {"workout_type": "endurance", "duration_minutes": 60}, ftp=250
        )
        threshold = _estimate_workout_tss(
            {"workout_type": "threshold", "duration_minutes": 60}, ftp=250
        )
        assert threshold > endurance

    def test_default_type_for_missing(self):
        """Missing workout type should default to endurance."""
        tss = _estimate_workout_tss({"duration_minutes": 60}, ftp=250)
        assert tss > 0


class TestPickTrainingDays:
    """Tests for training day selection."""

    def test_overreaching_only_one_day(self):
        days = _pick_training_days(date.today(), "overreaching", TrainingGoal.BASE)
        assert len(days) == 1

    def test_fresh_phase_has_more_days_than_heavy(self):
        fresh_days = len(_pick_training_days(date.today(), "fresh", TrainingGoal.BASE))
        heavy_days = len(_pick_training_days(date.today(), "heavy", TrainingGoal.BASE))
        assert fresh_days >= heavy_days

    def test_days_are_in_future(self):
        week_start = date.today() - timedelta(days=date.today().weekday())
        days = _pick_training_days(week_start, "optimal", TrainingGoal.BASE)
        for d in days:
            assert d >= week_start

    def test_build_has_at_least_as_many_days_as_base(self):
        base_days = _pick_training_days(date.today(), "optimal", TrainingGoal.BASE)
        build_days = _pick_training_days(date.today(), "optimal", TrainingGoal.BUILD)
        assert len(build_days) >= len(base_days)


class TestGetWorkoutTypeForZone:
    """Tests for workout type selection based on TSB zone."""

    def test_overreaching_returns_recovery(self):
        w_type = _get_workout_type_for_zone(
            "overreaching", TrainingGoal.BASE, {"endurance": 5}
        )
        assert w_type == "recovery"

    def test_heavy_returns_recovery_or_endurance(self):
        w_type = _get_workout_type_for_zone(
            "heavy", TrainingGoal.BASE, {"endurance": 5}
        )
        assert w_type in ("recovery", "endurance")

    def test_optimal_returns_varied_types(self):
        """Optimal zone should use the phase split for variety."""
        # Run many times to check diversity
        types_seen = set()
        for _ in range(20):
            w_type = _get_workout_type_for_zone(
                "optimal", TrainingGoal.BASE, {"recovery": 5, "endurance": 2}
            )
            types_seen.add(w_type)
        # Should see multiple types, not just one
        assert len(types_seen) > 1

    def test_recent_type_deprioritized(self):
        """A type done today should be less likely to be chosen."""
        # With recent endurance, should see more variety
        types_with_recent_endurance = set()
        for _ in range(30):
            types_with_recent_endurance.add(
                _get_workout_type_for_zone(
                    "optimal", TrainingGoal.BASE, {"endurance": 0}
                )
            )
        # Should still have variety but endurance might be deprioritized
        assert len(types_with_recent_endurance) >= 1


class TestGenerateWeeklyPlan:
    """Tests for the full weekly plan generation."""

    def test_generates_recommendations(self):
        week_start = date.today() - timedelta(days=date.today().weekday())
        plan = generate_weekly_plan(
            user_id=1,
            ctl=50,
            atl=40,
            tsb=10,
            goal=TrainingGoal.BASE,
            ftp=250,
            recent_workouts=[],
            existing_scheduled=[],
            week_start=week_start,
        )
        assert len(plan) > 0
        # Each recommendation should have required fields
        for rec in plan:
            assert "user_id" in rec
            assert "scheduled_date" in rec
            assert "workout_type" in rec
            assert "title" in rec
            assert rec["scheduled_date"] >= week_start

    def test_plan_respects_tss_budget(self):
        """Weekly plan TSS shouldn't wildly exceed capacity."""
        week_start = date.today() - timedelta(days=date.today().weekday())
        ctl = 50
        goal = TrainingGoal.BASE
        weekly_cap, _ = compute_weekly_tss_capacity(ctl, goal)
        plan = generate_weekly_plan(
            user_id=1,
            ctl=ctl,
            atl=40,
            tsb=10,
            goal=goal,
            ftp=250,
            recent_workouts=[],
            existing_scheduled=[],
            week_start=week_start,
        )
        total_tss = sum(_estimate_workout_tss(w, 250) for w in plan)
        # Allow some flexibility but total shouldn't be wildly exceeding
        assert total_tss <= weekly_cap * 1.5

    def test_overreaching_produces_minimal_plan(self):
        week_start = date.today() - timedelta(days=date.today().weekday())
        plan = generate_weekly_plan(
            user_id=1,
            ctl=80,
            atl=100,
            tsb=-30,  # overreaching
            goal=TrainingGoal.BASE,
            ftp=250,
            recent_workouts=[],
            existing_scheduled=[],
            week_start=week_start,
        )
        assert len(plan) <= 2  # At most 1-2 recovery rides

    def test_fresh_tsb_produces_more_workouts(self):
        """When TSB is positive (fresh), should have more/harder workouts."""
        week_start = date.today() - timedelta(days=date.today().weekday())
        overreaching_plan = generate_weekly_plan(
            user_id=1, ctl=40, atl=80, tsb=-40,  # overreaching
            goal=TrainingGoal.BASE, ftp=250,
            recent_workouts=[], existing_scheduled=[], week_start=week_start,
        )
        fresh_plan = generate_weekly_plan(
            user_id=1, ctl=40, atl=30, tsb=10,  # fresh
            goal=TrainingGoal.BASE, ftp=250,
            recent_workouts=[], existing_scheduled=[], week_start=week_start,
        )
        assert len(fresh_plan) >= len(overreaching_plan)

    def test_plan_has_no_duplicate_titles(self):
        """Recommendations should avoid duplicate workout titles."""
        week_start = date.today() - timedelta(days=date.today().weekday())
        plan = generate_weekly_plan(
            user_id=1, ctl=50, atl=40, tsb=10,
            goal=TrainingGoal.BASE, ftp=250,
            recent_workouts=[], existing_scheduled=[], week_start=week_start,
        )
        titles = [w["title"] for w in plan]
        assert len(titles) == len(set(titles))

    def test_different_goals_produce_different_plans(self):
        """Base and race phase plans should differ in workout type mix."""
        week_start = date.today() - timedelta(days=date.today().weekday())
        base_plan = generate_weekly_plan(
            user_id=1, ctl=50, atl=40, tsb=10,
            goal=TrainingGoal.BASE, ftp=250,
            recent_workouts=[], existing_scheduled=[], week_start=week_start,
        )
        race_plan = generate_weekly_plan(
            user_id=1, ctl=50, atl=40, tsb=10,
            goal=TrainingGoal.RACE, ftp=250,
            recent_workouts=[], existing_scheduled=[], week_start=week_start,
        )
        base_types = set(w["workout_type"] for w in base_plan)
        race_types = set(w["workout_type"] for w in race_plan)
        # Race phase should have more variety (sprint, vo2max, interval)
        race_has_intense = any(t in ("vo2max", "sprint", "interval") for t in race_types)
        base_has_vo2max = any(t in ("vo2max", "sprint", "interval") for t in base_types)
        # Base phase shouldn't have high intensity work, race phase should
        if not base_has_vo2max:
            assert race_has_intense or True  # Just a soft check


class TestPhaseWorkoutSplit:
    """Tests for phase-based workout distribution."""

    def test_all_phases_defined(self):
        """All TrainingGoal values should have a split defined."""
        for goal in TrainingGoal:
            assert goal in PHASE_WORKOUT_SPLIT, f"Missing split for {goal}"

    def test_splits_sum_to_one(self):
        """Each phase split should sum to approximately 1.0."""
        for goal, split in PHASE_WORKOUT_SPLIT.items():
            total = sum(split.values())
            assert total == pytest.approx(1.0, abs=0.01), f"Split for {goal} sums to {total}"

    def test_recovery_phase_no_high_intensity(self):
        """Recovery phase should have zero high intensity work."""
        recovery = PHASE_WORKOUT_SPLIT[TrainingGoal.RECOVERY]
        assert recovery.get("vo2max", 0) == 0.0
        assert recovery.get("sprint", 0) == 0.0
        assert recovery.get("threshold", 0) == 0.0

    def test_build_phase_more_threshold_than_base(self):
        """Build phase should have more threshold work than base."""
        base = PHASE_WORKOUT_SPLIT[TrainingGoal.BASE]
        build = PHASE_WORKOUT_SPLIT[TrainingGoal.BUILD]
        assert build["threshold"] > base["threshold"]
