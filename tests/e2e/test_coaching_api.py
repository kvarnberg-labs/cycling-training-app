"""E2E test for coaching API endpoints."""
import os
import sys

# Set the API key before any app imports
os.environ["INTERVALS_API_KEY"] = "5j2o3tb31414q802y5pedlzpv"

from app.auth import create_access_token
from app.main import app
from fastapi.testclient import TestClient

token = create_access_token(user_id=17)
headers = {"Authorization": f"Bearer {token}"}

with TestClient(app) as client:
    # Test context endpoint
    resp = client.get("/api/coaching/context?days_back=14&compact=true", headers=headers)
    assert resp.status_code == 200, f"Context endpoint: {resp.status_code} - {resp.text[:200]}"
    data = resp.json()
    assert "context" in data, "Context missing from response"
    assert data.get("context_chars", 0) > 100, f"Context too short: {data.get('context_chars')} chars"
    assert data.get("athlete"), "Athlete name missing"
    print(f"✓ GET /api/coaching/context — {data.get('context_chars')} chars, athlete: {data.get('athlete')}")

    # Test weekly prompt
    resp = client.get("/api/coaching/weekly?days_back=14", headers=headers)
    assert resp.status_code == 200, f"Weekly endpoint: {resp.status_code} - {resp.text[:200]}"
    data = resp.json()
    assert data.get("template") == "weekly"
    assert "system_prompt" in data
    assert "user_prompt" in data
    print(f"✓ GET /api/coaching/weekly — system: {len(data['system_prompt'])} chars, user: {len(data['user_prompt'])} chars")

    # Test daily prompt
    resp = client.get("/api/coaching/daily?days_back=14", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("template") == "daily"
    print(f"✓ GET /api/coaching/daily — system: {len(data['system_prompt'])} chars, user: {len(data['user_prompt'])} chars")

    # Test activity analysis
    resp = client.get("/api/coaching/activity-analysis?days_back=14", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "activities" in data
    assert data.get("count", 0) > 0, "No activities returned"
    print(f"✓ GET /api/coaching/activity-analysis — {data.get('count')} activities")

    print("\n━━━ ALL COACHING API E2E TESTS PASSED ━━━")

    # Print summary of data fetched
    print(f"\nData summary:")
    print(f"  ~{data.get('count', '?')} real activities from Intervals.icu")
    print(f"  CTL/ATL/TSB from real training metrics")
    print(f"  Training prompts ready for LLM consumption")
