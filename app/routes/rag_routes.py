from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates

from src.rag.rag_services import RAGService

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

rag_service = RAGService()


@router.get("/rag")
def rag_page(request: Request):

    return templates.TemplateResponse(
        "rag/rag.html",
        {
            "request": request,
            "answer": None,
            "message": None
        }
    )


@router.post("/rag/upload")
async def upload_pdf(
        request: Request,
        file: UploadFile = File(...)
):

    file_path = f"data/uploads/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    result = rag_service.ingest_pdf(file_path)

    return templates.TemplateResponse(
        "rag/rag.html",
        {
            "request": request,
            "answer": None,
            "message": f"PDF uploaded successfully. Chunks created: {result['chunks']}"
        }
    )


@router.post("/rag/ask")
def ask_rag(
        request: Request,
        question: str = Form(...)
):

    result = rag_service.ask(question)

    return templates.TemplateResponse(
        "rag/rag.html",
        {
            "request": request,
            "answer": result["answer"],
            "message": None
        }
    )