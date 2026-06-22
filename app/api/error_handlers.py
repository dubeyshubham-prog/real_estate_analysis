from fastapi import FastAPI
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from app.templates import templates
from src.common.exceptions import EstateAIError
from src.common.exceptions import ResourceNotFoundError
from src.monitoring.logging import get_logger


logger = get_logger(__name__)


def _wants_html(request: Request) -> bool:
    return "text/html" in request.headers.get("accept", "")


def register_error_handlers(app: FastAPI) -> None:
    """Register consistent JSON responses for expected application errors."""

    @app.exception_handler(EstateAIError)
    async def estate_ai_error_handler(
        request: Request,
        error: EstateAIError,
    ) -> JSONResponse:
        logger.warning(
            "Application error on %s %s: %s",
            request.method,
            request.url.path,
            error,
        )
        status_code = (
            404 if isinstance(error, ResourceNotFoundError) else 400
        )
        if _wants_html(request):
            return templates.TemplateResponse(
                request=request,
                name="error.html",
                status_code=status_code,
                context={
                    "title": "We could not complete that request",
                    "message": str(error),
                    "details": [],
                    "status_code": status_code,
                },
            )
        return JSONResponse(
            status_code=status_code,
            content={
                "error": error.__class__.__name__,
                "message": str(error),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        details = [
            (
                f"{str(item['loc'][-1]).replace('_', ' ').title() if item['loc'] else 'Input'}: "
                f"{item['msg']}"
            )
            for item in error.errors()
        ]
        if _wants_html(request):
            return templates.TemplateResponse(
                request=request,
                name="error.html",
                status_code=422,
                context={
                    "title": "Please check your input",
                    "message": "Some submitted values are invalid.",
                    "details": details,
                    "status_code": 422,
                },
            )
        return JSONResponse(
            status_code=422,
            content={
                "error": "RequestValidationError",
                "message": "Request validation failed",
                "details": jsonable_encoder(error.errors()),
            },
        )
