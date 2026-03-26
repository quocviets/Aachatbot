"""
FastAPI application factory — main entry point.

Run with:
    cd backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Path setup ───────────────────────────────────────────────────────────────
# Allow imports from project root (Inference/, Core/)
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
for _path in [_PROJECT_ROOT, _BACKEND_DIR]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from backend.core.config import get_settings
from backend.core.logger import get_logger
from backend.core.exceptions import AppException
from backend.database.connection import init_db
from backend.api.v1.router import router as v1_router

settings = get_settings()
logger = get_logger("app")


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + warm up AI model. Shutdown: nothing needed."""
    logger.info("🚀 Starting Plant Disease AI Backend v%s", settings.app_version)

    # Init database tables
    await init_db()
    logger.info("✅ Database initialized")

    # Warm up AI model (load into cache before first request)
    try:
        from Inference.pipeline import _model_manager  # noqa: PLC0415
        _model_manager.get_stage1_model()
        logger.info("✅ Stage 1 model warmed up on %s", _model_manager._cache.keys())
    except Exception as exc:
        logger.warning("⚠️  Model warm-up skipped: %s", exc)

    yield

    logger.info("👋 Shutting down")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="🌿 Plant Disease AI API",
        description="Backend API for AI-powered plant disease classification.",
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global exception handler ─────────────────────────────────────────────
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
        logger.warning("AppException [%d]: %s", exc.status_code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"status": "error", "message": exc.message},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Internal server error"},
        )

    # ── Static files (serve uploaded images) ────────────────────────────────
    uploads_dir = settings.resolved_upload_dir
    app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

    # ── Routes ───────────────────────────────────────────────────────────────
    app.include_router(v1_router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
    )
