from collections.abc import AsyncGenerator
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
from .event_models import WorkflowRunRequest
from .workflow_utils import (
    list_workflow_modules,
    get_workflow_module,
    get_all_workflow_metadata,
)

router = APIRouter()


@router.post("/workflows/{workflow_id}/run")
async def run_workflow(
    workflow_id: str, request: WorkflowRunRequest
) -> StreamingResponse:
    mod = get_workflow_module(workflow_id)
    if not mod or not hasattr(mod, "get_workflow"):
        available = list_workflow_modules()
        return StreamingResponse(
            status_code=404,
            content={
                "detail": f"Workflow '{workflow_id}' not found.",
                "available_workflows": available,
            },
        )
    orchestrator = mod.get_workflow()
    return StreamingResponse(
        orchestrator.execute(request),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/workflows/metadata")
async def all_workflow_metadata() -> JSONResponse:
    return JSONResponse(content=get_all_workflow_metadata())


@router.get("/workflows/{workflow_id}/metadata")
async def workflow_metadata(workflow_id: str) -> JSONResponse:
    mod = get_workflow_module(workflow_id)
    if not mod or not hasattr(mod, "get_metadata"):
        available = list_workflow_modules()
        raise HTTPException(
            status_code=404,
            detail={
                "msg": f"Workflow '{workflow_id}' not found.",
                "available_workflows": available,
            },
        )
    return JSONResponse(content=mod.get_metadata())


@router.get("/health")
async def health_check() -> JSONResponse:
    return JSONResponse(content={"status": "healthy"})


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup
    print("Starting AI Orchestration Service")
    yield
    # Shutdown
    print("Shutting down AI Orchestration Service")


router.lifespan_context = lifespan
