from collections.abc import AsyncGenerator
from fastapi import APIRouter, FastAPI
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from .event_models import WorkflowRunRequest
from .orchestrator import WorkflowOrchestrator

router = APIRouter()


@router.post("/workflows/{workflow_id}/run")
async def run_workflow(
    workflow_id: str, request: WorkflowRunRequest
) -> StreamingResponse:
    orchestrator = WorkflowOrchestrator(workflow_id)
    return StreamingResponse(
        orchestrator.execute(request),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/health")
async def health_check() -> dict:
    return {"status": "healthy"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup
    print("Starting AI Orchestration Service")
    yield
    # Shutdown
    print("Shutting down AI Orchestration Service")


router.lifespan_context = lifespan
