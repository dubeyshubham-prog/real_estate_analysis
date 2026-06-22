from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Form
from fastapi import Request
from fastapi.responses import HTMLResponse

from app.api.dependencies import get_recommendation_service
from app.schemas.recommendation import RecommendationInput
from app.services.recommendation_service import RecommendationService
from app.templates import templates


router = APIRouter(tags=["recommendation"])


@router.get("/recommendation", response_class=HTMLResponse)
def recommendation_page(
    request: Request,
    service: Annotated[
        RecommendationService,
        Depends(get_recommendation_service),
    ],
):
    return templates.TemplateResponse(
        request=request,
        name="recommendation/recommendation.html",
        context={
            "property_names": service.get_property_names(),
            "recommendations": None,
            "selected_property": None,
        },
    )


@router.post("/recommendation", response_class=HTMLResponse)
def get_recommendations(
    request: Request,
    property_name: Annotated[str, Form(min_length=1, max_length=200)],
    service: Annotated[
        RecommendationService,
        Depends(get_recommendation_service),
    ],
    top_n: Annotated[int, Form(ge=1, le=20)] = 5,
):
    form = RecommendationInput(
        property_name=property_name,
        top_n=top_n,
    )
    return templates.TemplateResponse(
        request=request,
        name="recommendation/recommendation.html",
        context={
            "property_names": service.get_property_names(),
            "recommendations": service.get_recommendations(
                property_name=form.property_name,
                top_n=form.top_n,
            ),
            "selected_property": form.property_name,
        },
    )
