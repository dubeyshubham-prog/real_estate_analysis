from typing import Annotated

from fastapi import APIRouter
from fastapi import File
from fastapi import Request
from fastapi import UploadFile
from fastapi.responses import HTMLResponse

from app.api.dependencies import get_vision_service
from app.api.uploads import validate_upload
from app.templates import templates
from config.settings import Config
from src.common.exceptions import FeatureUnavailableError


router = APIRouter(tags=["vision"])


@router.get("/deep-vision", response_class=HTMLResponse)
def deep_vision_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="vision/deep_vision.html",
        context={
            "result": None,
            "vision_enabled": Config.ENABLE_VISION,
        },
    )


@router.post("/deep-vision", response_class=HTMLResponse)
async def deep_vision_search(
    request: Request,
    image: Annotated[UploadFile, File(...)],
):
    if not Config.ENABLE_VISION:
        raise FeatureUnavailableError(
            "Deep Vision is disabled in this free deployment because its "
            "source image library is not included."
        )

    await validate_upload(
        image,
        allowed_content_types={
            "image/jpeg",
            "image/png",
            "image/webp",
        },
        allowed_extensions={".jpg", ".jpeg", ".png", ".webp"},
    )
    result = get_vision_service().find_similar_images(
        uploaded_file=image,
        top_k=5,
    )
    return templates.TemplateResponse(
        request=request,
        name="vision/deep_vision.html",
        context={"result": result, "vision_enabled": True},
    )
