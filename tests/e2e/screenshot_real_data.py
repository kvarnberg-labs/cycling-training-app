"""Playwright screenshots for JohanM's real data."""
import os, random, requests

os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "/opt/data/playwright-browsers"

from playwright.sync_api import sync_playwright

BASE = "http://localhost:8000"
EMAIL = "joey@kvarnberg.dev"
PASSWORD = "intervals123"


def login_api():
    r = requests.post(f"{BASE}/api/auth/login", json={"email": EMAIL, "password": PASSWORD})
    if r.status_code == 200:
        return r.json()["access_token"]
    r = requests.post(f"{BASE}/api/auth/register", json={
        "email": EMAIL, "password": PASSWORD, "name": "JohanM", "ftp": 284,
    })
    return r.json()["access_token"]


def run():
    token = login_api()
    print(f"✓ Logged in as {EMAIL}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # ── Dashboard ──
        print("\n--- Dashboard ---")
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.set_extra_http_headers({"Authorization": f"Bearer {token}"})
        page.goto(f"{BASE}/")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)
        page.screenshot(path="/tmp/pw_dashboard.png", full_page=True)
        assert "auth/login" not in page.url, f"Redirected: {page.url}"
        print("  ✅ Dashboard loaded")
        page.close()

        # ── Insights (power curve) ──
        print("\n--- Insights ---")
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.set_extra_http_headers({"Authorization": f"Bearer {token}"})
        page.goto(f"{BASE}/insights")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(5000)
        page.screenshot(path="/tmp/pw_insights.png", full_page=True)
        assert "auth/login" not in page.url
        print("  ✅ Insights/power curve loaded")
        page.close()

        # ── Settings ──
        print("\n--- Settings ---")
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.set_extra_http_headers({"Authorization": f"Bearer {token}"})
        page.goto(f"{BASE}/settings")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)
        page.screenshot(path="/tmp/pw_settings.png", full_page=True)
        assert "auth/login" not in page.url
        print("  ✅ Settings loaded")
        page.close()

        browser.close()
        print("\n✅ Screenshots saved:")
        print("  /tmp/pw_dashboard.png")
        print("  /tmp/pw_insights.png")
        print("  /tmp/pw_settings.png")


if __name__ == "__main__":
    run()
