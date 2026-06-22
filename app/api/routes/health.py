from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config.settings import Config


router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def readiness() -> JSONResponse:
    checks = {
        "price_model": Config.PRICE_MODEL_FILE.is_file(),
        "price_reference": Config.PRICE_REFERENCE_DATA_FILE.is_file(),
        "recommendation_model": Config.RECOMMENDATION_MODEL_FILE.is_file(),
        "templates": Config.TEMPLATES_DIR.is_dir(),
        "static": Config.STATIC_DIR.is_dir(),
        "database_runtime": Config.RUNTIME_DIR.is_dir(),
    }
    optional_features = {
        "vision_enabled": Config.ENABLE_VISION,
        "vision_model": Config.ROOM_CLASSIFIER_MODEL_FILE.is_file(),
        "visual_database": (
            Config.VISUAL_EMBEDDING_DATABASE_FILE.is_file()
        ),
    }
    ready = all(checks.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "checks": checks,
            "optional_features": optional_features,
        },
    )
