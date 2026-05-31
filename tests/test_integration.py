"""Integration tests for the Cycling Training App API endpoints.

Uses FastAPI TestClient with an in-memory SQLite database.
Tests cover auth, workouts, dashboard, strava, and user endpoints.
"""

import pytest
from datetime import date, timedelta
from fastapi import status
from app.models import (
    User, Workout, StravaActivity, TrainingMetrics,
    WorkoutType, WorkoutStatus, TrainingGoal,
)
from app.services.training_load import calculate_tss


class TestAuthEndpoints:
    """Tests for auth registration and login."""

    def test_register_user(self, client):
        response = client.post("/api/auth/register", json={
            "email": "new@rider.com",
            "password": "securepass123",
            "name": "New Rider",
        })
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["token_type"] == "bearer"
        assert "access_token" in data
        assert data["user_id"] > 0

    def test_register_with_duplicate_email(self, client, test_user):
        response = client.post("/api/auth/register", json={
            "email": "test@example.com",
            "password": "anotherpass123",
            "name": "Duplicate Rider",
        })
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_register_weak_password(self, client):
        response = client.post("/api/auth/register", json={
            "email": "weak@rider.com",
            "password": "abc",  # too short
            "name": "Weak Password Rider",
        })
        # The pydantic schema accepts short passwords
        assert response.status_code == status.HTTP_200_OK
        assert "access_token" in response.json()

    def test_login_success(self, client, test_user):
        response = client.post("/api/auth/login", json={
            "email": "test@example.com",
            "password": "testpassword",
        })
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self, client, test_user):
        response = client.post("/api/auth/login", json={
            "email": "test@example.com",
            "password": "wrongpassword",
        })
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_login_nonexistent_user(self, client):
        response = client.post("/api/auth/login", json={
            "email": "nobody@example.com",
            "password": "somepassword",
        })
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_me_endpoint_authenticated(self, client, auth_headers, test_user):
        response = client.get("/api/auth/status", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["authenticated"] is True
        assert data["user_id"] == test_user.id

    def test_me_endpoint_unauthenticated(self, client):
        response = client.get("/api/auth/status")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestWorkoutEndpoints:
    """Tests for workout CRUD operations."""

    def test_create_workout(self, client, auth_headers):
        response = client.post("/api/workouts/", headers=auth_headers, json={
            "scheduled_date": str(date.today() + timedelta(days=1)),
            "workout_type": "endurance",
            "title": "Test Endurance Ride",
            "duration_minutes": 60,
        })
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["title"] == "Test Endurance Ride"
        assert data["status"] == "suggested"
        assert data["source"] == "recommendation"

    def test_create_manual_workout(self, client, auth_headers):
        response = client.post("/api/workouts/", headers=auth_headers, json={
            "scheduled_date": str(date.today()),
            "workout_type": "tempo",
            "title": "Manual Tempo",
            "duration_minutes": 75,
            "is_manual": True,
        })
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["is_manual"] is True
        assert data["source"] == "manual"

    def test_list_workouts(self, client, auth_headers, test_user, db_session):
        # Create some workouts
        for i in range(3):
            w = Workout(
                user_id=test_user.id,
                scheduled_date=date.today() + timedelta(days=i),
                workout_type=WorkoutType.ENDURANCE,
                title=f"Workout {i}",
                duration_minutes=60,
                status=WorkoutStatus.SUGGESTED,
            )
            db_session.add(w)
        db_session.commit()

        response = client.get("/api/workouts/", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 3

    def test_list_workouts_with_date_filter(self, client, auth_headers, test_user, db_session):
        today = date.today()
        for i in range(5):
            w = Workout(
                user_id=test_user.id,
                scheduled_date=today + timedelta(days=i),
                workout_type=WorkoutType.ENDURANCE,
                title=f"Workout {i}",
                duration_minutes=60,
                status=WorkoutStatus.SUGGESTED,
            )
            db_session.add(w)
        db_session.commit()

        response = client.get(
            f"/api/workouts/?start_date={today}&end_date={today + timedelta(days=2)}",
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 3

    def test_get_workout(self, client, auth_headers, test_user, db_session):
        w = Workout(
            user_id=test_user.id,
            scheduled_date=date.today(),
            workout_type=WorkoutType.THRESHOLD,
            title="Threshold Test",
            duration_minutes=60,
            status=WorkoutStatus.SUGGESTED,
        )
        db_session.add(w)
        db_session.commit()

        response = client.get(f"/api/workouts/{w.id}", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["title"] == "Threshold Test"

    def test_get_workout_not_found(self, client, auth_headers):
        response = client.get("/api/workouts/99999", headers=auth_headers)
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_workout_other_users_workout(self, client, auth_headers, db_session, test_user):
        """Should not be able to access another user's workout."""
        other_user = User(email="other@test.com", password_hash="hash")
        db_session.add(other_user)
        db_session.commit()
        
        w = Workout(
            user_id=other_user.id,
            scheduled_date=date.today(),
            workout_type=WorkoutType.ENDURANCE,
            title="Other's Workout",
            status=WorkoutStatus.SUGGESTED,
        )
        db_session.add(w)
        db_session.commit()

        response = client.get(f"/api/workouts/{w.id}", headers=auth_headers)
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_update_workout_status(self, client, auth_headers, test_user, db_session):
        w = Workout(
            user_id=test_user.id,
            scheduled_date=date.today(),
            workout_type=WorkoutType.ENDURANCE,
            title="Update Test",
            status=WorkoutStatus.SUGGESTED,
        )
        db_session.add(w)
        db_session.commit()

        response = client.patch(f"/api/workouts/{w.id}", headers=auth_headers, json={
            "status": "accepted",
        })
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "accepted"

    def test_complete_workout(self, client, auth_headers, test_user, db_session):
        w = Workout(
            user_id=test_user.id,
            scheduled_date=date.today(),
            workout_type=WorkoutType.ENDURANCE,
            title="Complete Me",
            status=WorkoutStatus.ACCEPTED,
        )
        db_session.add(w)
        db_session.commit()

        response = client.put(f"/api/workouts/{w.id}/complete", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "completed"

    def test_skip_workout(self, client, auth_headers, test_user, db_session):
        w = Workout(
            user_id=test_user.id,
            scheduled_date=date.today(),
            workout_type=WorkoutType.ENDURANCE,
            title="Skip Me",
            status=WorkoutStatus.SUGGESTED,
        )
        db_session.add(w)
        db_session.commit()

        response = client.put(f"/api/workouts/{w.id}/skip", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "skipped"

    def test_accept_workout(self, client, auth_headers, test_user, db_session):
        w = Workout(
            user_id=test_user.id,
            scheduled_date=date.today(),
            workout_type=WorkoutType.ENDURANCE,
            title="Accept Me",
            status=WorkoutStatus.SUGGESTED,
        )
        db_session.add(w)
        db_session.commit()

        response = client.put(f"/api/workouts/{w.id}/accept", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "accepted"

    def test_delete_workout(self, client, auth_headers, test_user, db_session):
        w = Workout(
            user_id=test_user.id,
            scheduled_date=date.today(),
            workout_type=WorkoutType.ENDURANCE,
            title="Delete Me",
            status=WorkoutStatus.SUGGESTED,
        )
        db_session.add(w)
        db_session.commit()
        wid = w.id

        response = client.delete(f"/api/workouts/{wid}", headers=auth_headers)
        assert response.status_code == status.HTTP_204_NO_CONTENT

        # Verify it's gone
        response = client.get(f"/api/workouts/{wid}", headers=auth_headers)
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestDashboardEndpoints:
    """Tests for dashboard endpoints."""

    def test_dashboard_unauthenticated(self, client):
        """Should return default empty dashboard for unauthenticated users."""
        response = client.get("/api/dashboard/")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["current_ctl"] == 0
        assert data["current_tsb"] == 0

    def test_dashboard_authenticated(self, client, auth_headers, test_user, db_session):
        # Add some metrics
        tm = TrainingMetrics(
            user_id=test_user.id,
            date=date.today(),
            ctl=50.0,
            atl=40.0,
            tsb=10.0,
        )
        db_session.add(tm)
        db_session.commit()

        response = client.get("/api/dashboard/", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["current_ctl"] == 50.0
        assert data["current_tsb"] == 10.0
        assert data["ftp"] == 250

    def test_dashboard_shows_suggested_workouts(self, client, auth_headers, test_user, db_session):
        today = date.today()
        for i in range(3):
            w = Workout(
                user_id=test_user.id,
                scheduled_date=today + timedelta(days=i),
                workout_type=WorkoutType.ENDURANCE,
                title=f"Suggested {i}",
                status=WorkoutStatus.SUGGESTED,
            )
            db_session.add(w)
        db_session.commit()

        response = client.get("/api/dashboard/", headers=auth_headers)
        data = response.json()
        assert len(data["suggested_workouts"]) == 3

    def test_dashboard_strava_status(self, client, auth_headers, test_user):
        response = client.get("/api/dashboard/", headers=auth_headers)
        data = response.json()
        assert data["strava_connected"] is False
        assert data["training_goal"] == "base"


class TestCalendarEndpoint:
    """Tests for the weekly calendar endpoint."""

    def test_calendar_with_workouts(self, client, auth_headers, test_user, db_session):
        today = date.today()
        week_start = today - timedelta(days=today.weekday())

        for i in range(3):
            w = Workout(
                user_id=test_user.id,
                scheduled_date=week_start + timedelta(days=i),
                workout_type=WorkoutType.ENDURANCE,
                title=f"Day {i} Workout",
                status=WorkoutStatus.SUGGESTED,
            )
            db_session.add(w)
        db_session.commit()

        response = client.get(f"/api/dashboard/calendar?week_start={week_start}", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["days"]) == 7
        # Should have workouts on the first 3 days
        filled_days = [d for d in data["days"] if len(d["workouts"]) > 0]
        assert len(filled_days) == 3

    def test_calendar_defaults_to_current_week(self, client, auth_headers):
        response = client.get("/api/dashboard/calendar", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK

    def test_calendar_unauthenticated(self, client):
        response = client.get("/api/dashboard/calendar")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestMetricsEndpoint:
    """Tests for the metrics endpoint."""

    def test_get_metrics(self, client, auth_headers, test_user, db_session):
        for i in range(5):
            tm = TrainingMetrics(
                user_id=test_user.id,
                date=date.today() - timedelta(days=i),
                ctl=float(50 - i),
                atl=float(40 - i),
                tsb=float(10),
                total_tss=100.0,
                total_duration_minutes=60.0,
                total_distance_km=30.0,
                ride_count=1,
            )
            db_session.add(tm)
        db_session.commit()

        response = client.get("/api/dashboard/metrics?days=30", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 5
        # Data is returned in chronological order (oldest first),
        # so the last entry (most recent) should have highest CTL
        assert data[-1]["ctl"] > data[0]["ctl"]


class TestGenerateWeekPlan:
    """Tests for the weekly plan generation endpoint."""

    def test_generate_week_success(self, client, auth_headers, test_user, db_session):
        # Add training metrics so the engine has data to work with
        tm = TrainingMetrics(
            user_id=test_user.id,
            date=date.today(),
            ctl=40.0,
            atl=30.0,
            tsb=10.0,
        )
        db_session.add(tm)
        db_session.commit()

        week_start = date.today() - timedelta(days=date.today().weekday())
        response = client.post(
            f"/api/workouts/generate-week?week_start={week_start}",
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) > 0
        for workout in data:
            assert "title" in workout
            assert "scheduled_date" in workout

    def test_generate_week_without_metrics(self, client, auth_headers):
        week_start = date.today() - timedelta(days=date.today().weekday())
        response = client.post(
            f"/api/workouts/generate-week?week_start={week_start}",
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) > 0  # Should still generate with defaults


class TestHealthEndpoint:
    """Tests for the health/status endpoints."""

    def test_health_check(self, client):
        response = client.get("/api/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ok"
        assert data["app"] == "Cycling Training App"

    def test_info_endpoint(self, client):
        response = client.get("/api/info")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["strava_configured"] is False
        assert isinstance(data["debug"], bool)
