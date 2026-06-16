from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.services.vision_services import VisionService


router = APIRouter()

templates = Jinja2Templates(
    directory="app/templates"
)

vision_service = VisionService()

@router.get("/deep-vision", response_class=HTMLResponse)
def deep_vision_page(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="vision/deep_vision.html",
        context={
            "result": None
        }
    )

@router.post(
    "/deep-vision",
    response_class=HTMLResponse
)
def deep_vision_search(
        request: Request,
        image: UploadFile = File(...)
):

    result = (
        vision_service.find_similar_images(
            uploaded_file=image,
            top_k=5
        )
    )

    return templates.TemplateResponse(
        request=request,
        name="vision/deep_vision.html",
        context={
            "result": result
        }
    )