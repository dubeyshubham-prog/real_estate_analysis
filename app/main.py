from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes.recommendation_routes import router as recommendation_router
from app.routes.prediction_routes import router as prediction_router
from app.routes.analysis_routes import router as analysis_router


app = FastAPI(
    title="Real Estate Intelligence Platform"
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