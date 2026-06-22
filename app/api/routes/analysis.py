from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request
from fastapi.responses import HTMLResponse

from app.api.dependencies import get_analysis_service
from app.services.analysis_service import AnalysisService
from app.templates import templates


router = APIRouter(tags=["analysis"])


@router.get("/analysis", response_class=HTMLResponse)
def analysis_page(
    request: Request,
    service: Annotated[AnalysisService, Depends(get_analysis_service)],
):
    return templates.TemplateResponse(
        request=request,
        name="analysis/analysis.html",
        context={"charts": service.get_dashboard_charts()},
    )
