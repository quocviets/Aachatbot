"""
Centralized settings — loaded from .env via pydantic-settings.
All configuration must live here. Never import os.environ directly elsewhere.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), "..", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = False
    app_version: str = "1.0.0"

    # Storage
    upload_dir: str = "uploads"
    max_file_size_mb: int = 5

    # Database
    database_url: str = "sqlite+aiosqlite:///./plant_disease.db"

    # AI Model — empty = auto-detect from project root
    model_dir: str = ""

    # CORS
    allowed_origins: str = "*"

    # Rate limit
    rate_limit: str = "20/minute"

    # ─── Derived properties ──────────────────────────────────────────────────

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def allowed_origins_list(self) -> list[str]:
        if self.allowed_origins == "*":
            return ["*"]
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def resolved_model_dir(self) -> str:
        """Resolve model directory to absolute path (relative to project root)."""
        if self.model_dir:
            return os.path.abspath(self.model_dir)
        # Default: Model/ directory two levels up from backend/core/
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(project_root, "Model")

    @property
    def resolved_upload_dir(self) -> str:
        """Resolve upload directory relative to backend/."""
        backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(backend_root, self.upload_dir)
        os.makedirs(path, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
