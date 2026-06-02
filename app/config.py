"""Configuration for the cycling training app."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # App
    app_name: str = "Cycling Training App"
    debug: bool = True
    secret_key: str = "change-me-in-production-use-a-real-secret"

    # Database
    database_url: str = "sqlite:///./cycling_trainer.db"

    # Strava API
    strava_client_id: str = ""
    strava_client_secret: str = ""
    strava_redirect_uri: str = "http://localhost:8000/auth/strava/callback"

    # Base URL for the app
    base_url: str = "http://localhost:8000"

    # Training defaults
    default_ftp: int = 200  # watts
    default_weight: float = 75.0  # kg
    default_hr_rest: int = 60  # bpm
    default_hr_max: int = 185  # bpm

    # LLM for recommendations
    llm_api_key: str = ""
    llm_api_base: str = ""  # OpenAI-compatible endpoint (e.g. https://api.openai.com/v1)
    llm_model: str = "gpt-4o-mini"  # Model to use for recommendations
    llm_max_tokens: int = 4096

    # Intervals.icu
    intervals_api_key: str = ""
    intervals_api_base: str = "https://intervals.icu/api/v1"
    intervals_athlete_id: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
