from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.services.agent_services import AgentService


router = APIRouter()

templates = Jinja2Templates(
    directory="app/templates"
)

agent_service = AgentService()


@router.get("/ai-assistant", response_class=HTMLResponse)
def agent_page(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="agent/agent.html",
        context={
            "query": None,
            "result": None
        }
    )


@router.post("/ai-assistant", response_class=HTMLResponse)
def ask_agent(
        request: Request,
        query: str = Form(...)
):

    result = agent_service.ask_agent(
        query=query
    )

    return templates.TemplateResponse(
        request=request,
        name="agent/agent.html",
        context={
            "query": query,
            "result": result
        }
    )