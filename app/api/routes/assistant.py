from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Form
from fastapi import Request
from fastapi.responses import HTMLResponse

from app.api.dependencies import get_agent_service
from app.schemas.assistant import AssistantInput
from app.services.agent_services import AgentService
from app.templates import templates


router = APIRouter(tags=["assistant"])


@router.get("/ai-assistant", response_class=HTMLResponse)
def agent_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="agent/agent.html",
        context={"query": None, "result": None},
    )


@router.post("/ai-assistant", response_class=HTMLResponse)
def ask_agent(
    request: Request,
    query: Annotated[str, Form(min_length=3, max_length=1_000)],
    service: Annotated[AgentService, Depends(get_agent_service)],
):
    form = AssistantInput(query=query)
    return templates.TemplateResponse(
        request=request,
        name="agent/agent.html",
        context={
            "query": form.query,
            "result": service.ask_agent(query=form.query),
        },
    )
