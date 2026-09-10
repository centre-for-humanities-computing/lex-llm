from collections.abc import AsyncGenerator
import asyncio
import os
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
from .event_models import WorkflowRunRequest
from .workflow_utils import (
    list_workflow_modules,
    get_workflow_module,
    get_all_workflow_metadata,
)
from .observability.run_recorder import get_recorder
from .observability.logging_setup import get_logger, setup_logging

logger = get_logger()

router = APIRouter()

# Per-worker concurrency guard: rejects excess requests with 429 immediately
# rather than letting them queue and suffer long TTFT under load.
_MAX_CONCURRENT_WORKFLOWS = int(os.getenv("MAX_CONCURRENT_WORKFLOWS", "10"))
_workflow_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_WORKFLOWS)


async def _guarded_stream(
    inner: AsyncGenerator[str, None],
    semaphore: asyncio.Semaphore,
) -> AsyncGenerator[str, None]:
    """Yield from the inner generator, releasing the semaphore on completion
    or client disconnect so the slot is freed for the next request."""
    try:
        async for chunk in inner:
            yield chunk
    finally:
        semaphore.release()


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

    # Fast-fail: reject immediately if all workflow slots are taken.
    # The check + acquire is safe without a lock because there is no
    # ``await`` between them, so no other coroutine can interleave.
    if _workflow_semaphore._value <= 0:  # noqa: SLF001
        logger.warning(
            "request_rejected reason=at_capacity workflow=%s conversation_id=%s "
            "limit=%d",
            workflow_id,
            request.conversation_id,
            _MAX_CONCURRENT_WORKFLOWS,
        )
        raise HTTPException(
            status_code=429,
            detail="Server is at capacity. Please try again later.",
            headers={"Retry-After": "5"},
        )
    await _workflow_semaphore.acquire()

    try:
        metadata = mod.get_metadata()
        # Error out if the workflow is marked as inactive. This allows us to keep the workflow code in the repo for reference or future reactivation, but prevent it from being used in production.
        if metadata.get("status") == "inactive":
            raise HTTPException(
                status_code=503,
                detail=f"Workflow '{workflow_id}' is currently inactive.",
            )

        orchestrator = mod.get_workflow(request)
        orchestrator.workflow_id = workflow_id
    except Exception:
        # Logging only -- re-raised unchanged. Without this the client gets a
        # bare "Internal Server Error" and the cause is lost entirely.
        logger.exception(
            "workflow_setup_failed workflow=%s conversation_id=%s",
            workflow_id,
            request.conversation_id,
        )
        raise
    return StreamingResponse(
        _guarded_stream(orchestrator.execute(), _workflow_semaphore),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache"},
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Start the RunRecorder on boot, drain on shutdown."""
    setup_logging()
    logger.info("service_start max_concurrent_workflows=%d", _MAX_CONCURRENT_WORKFLOWS)
    recorder = get_recorder()
    await recorder.start()
    yield
    await recorder.stop()


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


router.lifespan_context = lifespan
