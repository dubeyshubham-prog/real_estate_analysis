from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.services.analysis_service import AnalysisService


router = APIRouter()

templates = Jinja2Templates(
    directory="app/templates"
)

analysis_service = AnalysisService()


# ==========================================================
# ANALYSIS DASHBOARD PAGE
# ==========================================================
@router.get("/analysis", response_class=HTMLResponse)
def analysis_page(request: Request):

    charts = analysis_service.get_dashboard_charts()

    return templates.TemplateResponse(
        request=request,
        name="analysis/analysis.html",
        context={
            "charts": charts
        }
    )