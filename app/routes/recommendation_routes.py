from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.services.recommendation_service import RecommendationService


router = APIRouter()

templates = Jinja2Templates(
    directory="app/templates"
)

recommendation_service = RecommendationService()


# ==========================================================
# RECOMMENDATION PAGE
# ==========================================================
@router.get("/recommendation", response_class=HTMLResponse)
def recommendation_page(request: Request):

    property_names = recommendation_service.get_property_names()

    return templates.TemplateResponse(
        request=request,
        name="recommendation/recommendation.html",
        context={
            "property_names": property_names,
            "recommendations": None,
            "selected_property": None
        }
    )


# ==========================================================
# RECOMMENDATION RESULT
# ==========================================================
@router.post("/recommendation", response_class=HTMLResponse)
def get_recommendations(
        request: Request,
        property_name: str = Form(...),
        top_n: int = Form(5)
):

    property_names = recommendation_service.get_property_names()

    recommendations = recommendation_service.get_recommendations(
        property_name=property_name,
        top_n=top_n
    )

    return templates.TemplateResponse(
        request=request,
        name="recommendation/recommendation.html",
        context={
            "property_names": property_names,
            "recommendations": recommendations,
            "selected_property": property_name
        }
    )