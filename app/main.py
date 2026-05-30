"""Main application entry point for the Cycling Training App API.

Run with: uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Depends
from sqlalchemy import func

from app.config import settings
from app.database import init_db, SessionLocal
from app.routers import strava, workouts, dashboard, user, auth, analytics
from app.services.metrics_compute import compute_all_users_metrics
from app.auth import optional_current_user, get_current_user
from app.models import User

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize DB on startup, cleanup on shutdown."""
    logger.info("Initializing database...")
    init_db()

    # Run daily metrics computation on startup
    try:
        logger.info("Computing training metrics...")
        compute_all_users_metrics()
    except Exception as e:
        logger.warning(f"Could not compute metrics on startup: {e}")

    # Check if Strava is configured
    if not settings.strava_client_id or not settings.strava_client_secret:
        logger.warning(
            "Strava API not configured! Set STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET "
            "environment variables or in .env file."
        )

    yield
    logger.info("Shutting down...")


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="A self-hosted cycling training app powered by Strava data. "
                "Analyzes training load and recommends workouts.",
    lifespan=lifespan,
)

# Mount static files
import os
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

# Include API routers
app.include_router(strava.router, prefix="/api")
app.include_router(workouts.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(user.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(auth.router, prefix="/api")


# ── Web UI Routes ──

@app.get("/")
async def index(
    request: Request,
    current_user: Optional[User] = Depends(optional_current_user),
):
    """Main dashboard page."""
    if not current_user:
        return RedirectResponse(url="/auth/login", status_code=302)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": settings.app_name,
        "strava_client_id": settings.strava_client_id,
        "strava_redirect_uri": settings.strava_redirect_uri,
    })


@app.get("/calendar")
async def calendar_view(
    request: Request,
    current_user: Optional[User] = Depends(optional_current_user),
):
    """Weekly calendar page."""
    if not current_user:
        return RedirectResponse(url="/auth/login", status_code=302)
    return templates.TemplateResponse("calendar.html", {
        "request": request,
        "title": "Calendar — " + settings.app_name,
    })


@app.get("/settings")
async def settings_view(
    request: Request,
    current_user: Optional[User] = Depends(optional_current_user),
):
    """Settings page."""
    if not current_user:
        return RedirectResponse(url="/auth/login", status_code=302)
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "title": "Settings — " + settings.app_name,
    })


@app.get("/pmc")
async def pmc_view(
    request: Request,
    current_user: Optional[User] = Depends(optional_current_user),
):
    """Performance Management Chart page."""
    if not current_user:
        return RedirectResponse(url="/auth/login", status_code=302)
    return templates.TemplateResponse("pmc.html", {
        "request": request,
        "title": "PMC — " + settings.app_name,
    })


@app.get("/insights")
async def insights_view(
    request: Request,
    current_user: Optional[User] = Depends(optional_current_user),
):
    """Workout history analytics / insights page."""
    if not current_user:
        return RedirectResponse(url="/auth/login", status_code=302)
    return templates.TemplateResponse("insights.html", {
        "request": request,
        "title": "Insights — " + settings.app_name,
    })


@app.exception_handler(404)
async def not_found(request: Request, exc):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": settings.app_name,
    }, status_code=404)


# ── Health / Info ──

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "app": settings.app_name, "version": "1.0.0"}


@app.get("/api/info")
def app_info():
    """Get app configuration info (no secrets)."""
    return {
        "app_name": settings.app_name,
        "strava_configured": bool(settings.strava_client_id),
        "debug": settings.debug,
    }
