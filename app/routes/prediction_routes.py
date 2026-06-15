from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.services.prediction_service import PredictionService


router = APIRouter()

templates = Jinja2Templates(
    directory="app/templates"
)

prediction_service = PredictionService()


# ==========================================================
# HOME PAGE
# ==========================================================
@router.get("/", response_class=HTMLResponse)
def home(request: Request):

    options = prediction_service.get_form_options()

    return templates.TemplateResponse(
        request=request,
        name="prediction/prediction.html",
        context={
            "options": options,
            "prediction": None
        }
    )


# ==========================================================
# PRICE PREDICTION ROUTE
# ==========================================================
@router.post("/predict", response_class=HTMLResponse)
def predict_price(
        request: Request,
        property_type: str = Form(...),
        sector: str = Form(...),
        bedRoom: float = Form(...),
        bathroom: float = Form(...),
        balcony: str = Form(...),
        agePossession: str = Form(...),
        floor_num: float = Form(...),
        total_floors: float = Form(...),
        built_up_area: float = Form(...),
        servant_room: int = Form(...),
        store_room: int = Form(...),
        furnishing_type: str = Form(...),
        luxury_category: str = Form(...),
        floor_category: str = Form(...)
):

    input_data = {
        "property_type": property_type,
        "sector": sector,
        "bedRoom": bedRoom,
        "bathroom": bathroom,
        "balcony": balcony,
        "agePossession": agePossession,
        "floor_num": floor_num,
        "total_floors": total_floors,
        "built_up_area": built_up_area,
        "servant room": servant_room,
        "store room": store_room,
        "furnishing_type": furnishing_type,
        "luxury_category": luxury_category,
        "floor_category": floor_category
    }

    prediction = prediction_service.predict_price(input_data)

    options = prediction_service.get_form_options()

    return templates.TemplateResponse(
        request=request,
        name="prediction/prediction.html",
        context={
            "options": options,
            "prediction": prediction
        }
    )