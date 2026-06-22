from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.error_handlers import register_error_handlers
from app.api.routes.analysis import router as analysis_router
from app.api.routes.assistant import router as assistant_router
from app.api.routes.health import router as health_router
from app.api.routes.prediction import router as prediction_router
from app.api.routes.rag import router as rag_router
from app.api.routes.recommendation import router as recommendation_router
from app.api.routes.vision import router as vision_router
from config.settings import Config
from src.database.initialization import initialize_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    Config.initialize_directories()
    initialize_database()
    yield


def create_app() -> FastAPI:
    """Create and configure the PropLens FastAPI application."""
    application = FastAPI(
        title=Config.APP_NAME,
        version="1.0.0",
        debug=Config.DEBUG,
        lifespan=lifespan,
    )
    application.add_middleware(GZipMiddleware, minimum_size=1_000)
    application.mount(
        "/static",
        StaticFiles(directory=str(Config.STATIC_DIR)),
        name="static",
    )
    application.mount(
        "/property-images",
        StaticFiles(
            directory=str(Config.PROPERTY_IMAGES_DIR),
            check_dir=False,
        ),
        name="property-images",
    )

    for router in (
        prediction_router,
        recommendation_router,
        analysis_router,
        vision_router,
        assistant_router,
        rag_router,
        health_router,
    ):
        application.include_router(router)

    register_error_handlers(application)
    return application


app = create_app()
