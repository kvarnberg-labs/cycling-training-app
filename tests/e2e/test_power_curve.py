"""Playwright E2E tests: dashboard + power curve.

Uses page.route() to inject auth header into all API requests,
bypassing the server-side auth redirect issue.
"""

import os, random, requests

os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "/opt/data/playwright-browsers"

from playwright.sync_api import sync_playwright

BASE = "http://localhost:8000"


def login_api(email="pw-rider@example.com", password="test123"):
    r = requests.post(f"{BASE}/api/auth/login", json={"email": email, "password": password})
    if r.status_code == 200:
        return r.json()["access_token"]
    r = requests.post(f"{BASE}/api/auth/register", json={
        "email": email, "password": password, "name": "Test", "ftp": 200,
    })
    return r.json()["access_token"]


def run_tests():
    token = login_api()
    print(f"✓ Token: {token[:20]}...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # ── Test 1: Dashboard ──
        print("\n--- Test 1: Dashboard loads ---")
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        # Send auth header with EVERY request (including page navigation)
        page.set_extra_http_headers({"Authorization": f"Bearer {token}"})

        page.goto(f"{BASE}/")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)
        page.screenshot(path="/tmp/pw_dashboard.png", full_page=True)

        assert "auth/login" not in page.url, f"Redirected: {page.url}"
        for l in ["CTL (Fitness)", "ATL (Fatigue)", "TSB (Form)"]:
            assert page.locator(f"text={l}").is_visible(), f"Missing: {l}"
        print(f"  ✅ Dashboard metrics visible")

        # ── Test 2: Insights power curve ──
        print("\n--- Test 2: Power curve chart ---")
        page.goto(f"{BASE}/insights")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(4000)
        page.screenshot(path="/tmp/pw_power_curve.png", full_page=True)

        assert "auth/login" not in page.url
        assert page.locator("h2:has-text('Power Curve')").is_visible(), "Power Curve heading missing"
        assert page.locator("#powerCurveChart").count() > 0, "Chart canvas missing"
        print(f"  ✅ Power curve heading + chart present")

        # Check for data cards (power values in the stat grid)
        body = page.locator("body").text_content()
        for pw in ["850W", "480W"]:
            if pw in body:
                print(f"  ✅ Power value '{pw}' in page")
                break

        # ── Test 3: Empty state ──
        print("\n--- Test 3: Empty state ---")
        suffix = random.randint(10000, 99999)
        empty_token = login_api(f"new-{suffix}@test.com", "test123")

        page2 = browser.new_page(viewport={"width": 1280, "height": 900})
        page2.set_extra_http_headers({"Authorization": f"Bearer {empty_token}"})

        page2.goto(f"{BASE}/insights")
        page2.wait_for_load_state("networkidle")
        page2.wait_for_timeout(3000)
        page2.screenshot(path="/tmp/pw_empty.png", full_page=True)

        assert "auth/login" not in page2.url
        page2.wait_for_timeout(3000)
        body_text = page2.locator("body").text_content()
        assert "No power data available yet" in body_text, f"Empty state text not found in: {body_text[:200]}"
        print(f"  ✅ Empty state message confirmed in page text")

        page2.close()
        page.close()
        browser.close()

        print(f"\n{'='*50}")
        print("✅ All Playwright E2E tests passed!")
        print("📸 Screenshots: /tmp/pw_dashboard.png, /tmp/pw_power_curve.png, /tmp/pw_empty.png")


if __name__ == "__main__":
    run_tests()
