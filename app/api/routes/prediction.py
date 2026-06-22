from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Form
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse
from pydantic import ValidationError

from app.api.dependencies import get_prediction_service
from app.schemas.prediction import PredictionInput
from app.services.prediction_service import PredictionService
from app.templates import templates


router = APIRouter(tags=["prediction"])


def prediction_form(
    property_type: Annotated[str, Form(min_length=1)],
    sector: Annotated[str, Form(min_length=1)],
    bedRoom: Annotated[float, Form(gt=0, le=20)],
    bathroom: Annotated[float, Form(gt=0, le=20)],
    balcony: Annotated[float, Form(ge=0, le=10)],
    agePossession: Annotated[str, Form(min_length=1)],
    floor_num: Annotated[float, Form(ge=-2, le=100)],
    total_floors: Annotated[float, Form(ge=0, le=100)],
    built_up_area: Annotated[float, Form(gt=100, le=100_000)],
    servant_room: Annotated[int, Form(ge=0, le=1)],
    store_room: Annotated[int, Form(ge=0, le=1)],
    furnishing_type: Annotated[str, Form(min_length=1)],
    luxury_category: Annotated[str, Form(min_length=1)],
    floor_category: Annotated[str, Form(min_length=1)],
) -> PredictionInput:
    try:
        return PredictionInput(
            property_type=property_type,
            sector=sector,
            bedRoom=bedRoom,
            bathroom=bathroom,
            balcony=balcony,
            agePossession=agePossession,
            floor_num=floor_num,
            total_floors=total_floors,
            built_up_area=built_up_area,
            servant_room=servant_room,
            store_room=store_room,
            furnishing_type=furnishing_type,
            luxury_category=luxury_category,
            floor_category=floor_category,
        )
    except ValidationError as error:
        # Convert schema-level checks into FastAPI's standard 422 response.
        raise RequestValidationError(
            error.errors(include_context=False)
        ) from error


@router.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    service: Annotated[PredictionService, Depends(get_prediction_service)],
):
    return templates.TemplateResponse(
        request=request,
        name="prediction/prediction.html",
        context={
            "options": service.get_form_options(),
            "prediction": None,
            "prediction_range": None,
        },
    )


@router.post("/predict", response_class=HTMLResponse)
def predict_price(
    request: Request,
    form: Annotated[PredictionInput, Depends(prediction_form)],
    service: Annotated[PredictionService, Depends(get_prediction_service)],
):
    prediction = service.predict_price(form.to_model_input())
    return templates.TemplateResponse(
        request=request,
        name="prediction/prediction.html",
        context={
            "options": service.get_form_options(),
            "prediction": prediction["predicted_price_cr"],
            "prediction_range": prediction,
        },
    )
