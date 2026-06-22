from typing import Annotated

from fastapi import APIRouter
from fastapi import File
from fastapi import Form
from fastapi import Request
from fastapi import UploadFile

from app.api.dependencies import get_rag_service
from app.api.uploads import safe_upload_path
from app.api.uploads import validate_upload
from app.templates import templates
from config.settings import Config


router = APIRouter(tags=["rag"])


@router.get("/rag")
def rag_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="rag/rag.html",
        context={
            "answer": None,
            "message": None,
            "max_upload_size_mb": Config.MAX_UPLOAD_SIZE_MB,
        },
    )


@router.post("/rag/upload")
async def upload_pdf(
    request: Request,
    file: Annotated[UploadFile, File(...)],
):
    content = await validate_upload(
        file,
        allowed_content_types={"application/pdf"},
        allowed_extensions={".pdf"},
    )
    file_path = safe_upload_path(
        Config.DOCUMENT_UPLOADS_DIR,
        file.filename or "document.pdf",
    )
    file_path.write_bytes(content)
    result = get_rag_service().ingest_pdf(str(file_path))
    return templates.TemplateResponse(
        request=request,
        name="rag/rag.html",
        context={
            "answer": None,
            "message": (
                "PDF uploaded successfully. "
                f"Chunks created: {result['chunks']}"
            ),
            "max_upload_size_mb": Config.MAX_UPLOAD_SIZE_MB,
        },
    )


@router.post("/rag/ask")
def ask_rag(
    request: Request,
    question: Annotated[str, Form(min_length=3, max_length=1_000)],
):
    result = get_rag_service().ask(question.strip())
    return templates.TemplateResponse(
        request=request,
        name="rag/rag.html",
        context={
            "answer": result["answer"],
            "message": None,
            "max_upload_size_mb": Config.MAX_UPLOAD_SIZE_MB,
        },
    )
