from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes.recommendation_routes import router as recommendation_router
from app.routes.prediction_routes import router as prediction_router
from app.routes.analysis_routes import router as analysis_router
from app.routes.vision_routes import router as vision_router
from fastapi.staticfiles import StaticFiles
from app.routes.agent_routes import router as agent_router
from app.routes.rag_routes import router as rag_router


app = FastAPI(
    title="Real Estate Intelligence Platform"
)
app.mount(
    "/property-images",
    StaticFiles(
        directory="data/property_images/raw/room-dataset"
    ),
    name="property-images"
)

# ==========================================================
# STATIC FILES
# ==========================================================
app.mount(
    "/static",
    StaticFiles(directory="app/static"),
    name="static"
)

# ==========================================================
# ROUTES
# ==========================================================
app.include_router(
    prediction_router
)

app.include_router(
    recommendation_router
)

app.include_router(
    analysis_router
)

app.include_router(
    vision_router
)

app.include_router(agent_router)
app.include_router(rag_router)