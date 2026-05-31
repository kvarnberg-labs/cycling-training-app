"""Unit tests for the training load engine (app/services/training_load.py).

Tests TSS calculation, CTL/ATL/TSB computation, workout type classification,
and the PMC series generator.
"""

import pytest
from datetime import date, timedelta
from app.services.training_load import (
    calculate_tss,
    exp_weighted_avg,
    estimate_tss_from_hr,
    estimate_tss_from_rpe,
    classify_workout_type,
    pmc_series,
    TAU_CTL,
    TAU_ATL,
)


class TestExpWeightedAvg:
    """Tests for the exponentially weighted moving average calculation."""

    def test_initial_value(self):
        """When previous is 0, result should converge toward current_value."""
        result = exp_weighted_avg(0.0, 100.0, TAU_CTL)
        alpha = 1 - __import__("math").exp(-1 / TAU_CTL)
        expected = 0 + (100 - 0) * alpha
        assert result == pytest.approx(expected, rel=1e-6)

    def test_steady_state(self):
        """When previous equals current_value, result should stay the same."""
        assert exp_weighted_avg(50.0, 50.0, TAU_CTL) == pytest.approx(50.0)

    def test_convergence_toward_value(self):
        """Repeated calls should converge toward the input value."""
        value = 0.0
        for _ in range(500):
            value = exp_weighted_avg(value, 100.0, TAU_CTL)
        assert value == pytest.approx(100.0, abs=2.0)

    def test_small_tau(self):
        """Smaller tau means faster adaptation."""
        slow = exp_weighted_avg(50.0, 100.0, 42)
        fast = exp_weighted_avg(50.0, 100.0, 7)
        assert fast > slow


class TestCalculateTSS:
    """Tests for Training Stress Score calculation."""

    def test_basic_tss(self):
        """Test standard TSS calculation with typical values."""
        # 200W NP, 250 FTP, 3600 seconds (1 hour)
        # IF = 200/250 = 0.8
        # TSS = (3600 * 200 * 0.8) / (250 * 3600) * 100 = 64.0
        tss = calculate_tss(normalized_power=200, ftp=250, duration_seconds=3600)
        expected = 64.0
        assert tss == pytest.approx(expected, abs=0.1)

    def test_zero_ftp_returns_zero(self):
        assert calculate_tss(200, 0, 3600) == 0.0

    def test_zero_duration_returns_zero(self):
        assert calculate_tss(200, 250, 0) == 0.0

    def test_zero_power_returns_zero(self):
        assert calculate_tss(0, 250, 3600) == 0.0

    def test_high_intensity_short_duration(self):
        """A hard 30-min threshold effort."""
        # 300W NP, 280 FTP, 1800 seconds
        # IF = 300/280 ≈ 1.071
        # TSS ≈ (1800 * 300 * 1.071) / (280 * 3600) * 100 ≈ 57.4
        tss = calculate_tss(normalized_power=300, ftp=280, duration_seconds=1800)
        assert tss > 50
        assert tss < 65

    def test_tss_scales_with_duration(self):
        """Doubling duration should roughly double TSS."""
        short = calculate_tss(200, 250, 1800)
        long = calculate_tss(200, 250, 3600)
        assert long == pytest.approx(short * 2, rel=0.01)

    def test_tss_returns_float(self):
        tss = calculate_tss(200, 250, 3600)
        assert isinstance(tss, float)


class TestEstimateTSSFromHR:
    """Tests for HR-based TSS estimation."""

    def test_basic_hr_estimate(self):
        """A moderate effort should produce a reasonable TSS estimate."""
        tss = estimate_tss_from_hr(
            avg_hr=145, resting_hr=55, max_hr=190,
            duration_minutes=60, ftp=250, weight_kg=72
        )
        assert tss > 0
        assert tss < 200

    def test_hr_below_resting_returns_zero(self):
        tss = estimate_tss_from_hr(
            avg_hr=50, resting_hr=55, max_hr=190,
            duration_minutes=60, ftp=250, weight_kg=72
        )
        assert tss == 0.0

    def test_max_hr_equals_resting_returns_zero(self):
        tss = estimate_tss_from_hr(
            avg_hr=60, resting_hr=60, max_hr=60,
            duration_minutes=60, ftp=250, weight_kg=72
        )
        assert tss == 0.0

    def test_higher_hr_same_duration_gives_higher_tss(self):
        """All else equal, higher HR should give higher TSS."""
        easy = estimate_tss_from_hr(120, 55, 190, 60, 250, 72)
        hard = estimate_tss_from_hr(160, 55, 190, 60, 250, 72)
        assert hard > easy


class TestEstimateTSSFromRPE:
    """Tests for RPE-based TSS estimation."""

    def test_basic_rpe_estimate(self):
        tss = estimate_tss_from_rpe(rpe=7, duration_minutes=60, ftp=250, weight_kg=72)
        # sRPE = 60 * 7 = 420, estimated TSS = 420 * 1.5 = 630
        assert tss > 0
        assert tss < 1000

    def test_zero_rpe_returns_zero(self):
        assert estimate_tss_from_rpe(0, 60, 250, 72) == 0.0

    def test_zero_duration_returns_zero(self):
        assert estimate_tss_from_rpe(5, 0, 250, 72) == 0.0

    def test_longer_duration_higher_tss(self):
        short = estimate_tss_from_rpe(5, 30, 250, 72)
        long = estimate_tss_from_rpe(5, 60, 250, 72)
        assert long > short


class TestClassifyWorkoutType:
    """Tests for workout type classification from power data."""

    def test_recovery(self):
        """IF < 0.55 → recovery."""
        result = classify_workout_type(
            normalized_power=100, ftp=250,
            average_hr=None, max_hr=None, resting_hr=None
        )
        assert result == "recovery"

    def test_endurance(self):
        """0.56-0.75 FTP → endurance."""
        result = classify_workout_type(
            normalized_power=160, ftp=250,
            average_hr=None, max_hr=None, resting_hr=None
        )
        assert result == "endurance"

    def test_tempo(self):
        """0.76-0.87 FTP → tempo."""
        result = classify_workout_type(
            normalized_power=200, ftp=250,
            average_hr=None, max_hr=None, resting_hr=None
        )
        assert result == "tempo"

    def test_threshold(self):
        """0.88-1.05 FTP → threshold."""
        result = classify_workout_type(
            normalized_power=240, ftp=250,
            average_hr=None, max_hr=None, resting_hr=None
        )
        assert result == "threshold"

    def test_vo2max(self):
        """1.06-1.20 FTP → vo2max."""
        result = classify_workout_type(
            normalized_power=285, ftp=250,
            average_hr=None, max_hr=None, resting_hr=None
        )
        assert result == "vo2max"

    def test_sprint(self):
        """>1.20 FTP → sprint."""
        result = classify_workout_type(
            normalized_power=320, ftp=250,
            average_hr=None, max_hr=None, resting_hr=None
        )
        assert result == "sprint"

    def test_default_for_zero_ftp(self):
        result = classify_workout_type(
            normalized_power=200, ftp=0,
            average_hr=None, max_hr=None, resting_hr=None
        )
        assert result == "endurance"

    def test_boundary_between_tempo_and_threshold(self):
        """Test around the 87-88% boundary."""
        tempo = classify_workout_type(
            normalized_power=217, ftp=250,  # 86.8% — should be tempo
            average_hr=None, max_hr=None, resting_hr=None
        )
        threshold = classify_workout_type(
            normalized_power=220, ftp=250,  # 88% — should be threshold
            average_hr=None, max_hr=None, resting_hr=None
        )
        assert tempo == "tempo"
        assert threshold == "threshold"


class TestPMCSeries:
    """Tests for Performance Management Chart time series."""

    def test_empty_input(self):
        series = pmc_series({}, num_days=30)
        assert series == []

    def test_single_day(self):
        daily_tss = {date.today(): 100.0}
        series = pmc_series(daily_tss, num_days=30)
        assert len(series) == 1  # Should only cover the date range with data
        assert len(series[0]) == 4  # (date, ctl, atl, tsb)

    def test_ctl_starts_low_and_builds(self):
        """With daily training, CTL should build up."""
        today = date.today()
        daily_tss = {}
        for i in range(14):
            daily_tss[today - timedelta(days=13 - i)] = 80.0
        series = pmc_series(daily_tss, num_days=14)
        assert len(series) == 14
        # CTL should be increasing over 14 days of consistent training
        first_ctl = series[0][1]
        last_ctl = series[-1][1]
        assert last_ctl > first_ctl

    def test_tsb_is_ctl_minus_atl(self):
        today = date.today()
        daily_tss = {today - timedelta(days=i): 100.0 for i in range(3)}
        series = pmc_series(daily_tss, num_days=3)
        for _, ctl, atl, tsb in series:
            assert tsb == pytest.approx(ctl - atl, abs=0.2)

    def test_zero_tss_maintains_decay(self):
        """Zero TSS days should cause CTL and ATL to decay."""
        today = date.today()
        # Start with a training day, then rest
        daily_tss = {
            today - timedelta(days=3): 100.0,
            today - timedelta(days=2): 0.0,
            today - timedelta(days=1): 0.0,
        }
        series = pmc_series(daily_tss)
        if len(series) >= 3:
            # CTL should decay during rest days
            ctl_day2 = series[-2][1]  # first rest day
            ctl_day3 = series[-1][1]  # second rest day
            assert ctl_day3 <= ctl_day2
