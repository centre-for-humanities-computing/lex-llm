from typing import AsyncGenerator, Any, Dict, List
import pytest
import pytest_asyncio
from unittest.mock import patch
from lex_llm.api.orchestrator import Orchestrator
from lex_llm.api.event_emitter import EventEmitter
from lex_llm.api.event_models import WorkflowRunRequest, ConversationMessage


@pytest_asyncio.fixture
async def dummy_request() -> WorkflowRunRequest:
    return WorkflowRunRequest(
        conversation_id="conv-1",
        user_input="Hello?",
        conversation_history=[ConversationMessage(role="user", content="Hi!")],
    )


@pytest.mark.asyncio
async def test_orchestrator_executes_steps_in_order(
    dummy_request: WorkflowRunRequest,
) -> None:
    calls: List[str] = []

    async def step1(
        context: Dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str, None]:
        calls.append("step1")
        context["step1"] = True
        yield "event1"

    async def step2(
        context: Dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str, None]:
        calls.append("step2")
        context["step2"] = True
        yield "event2"

    orch: Orchestrator = Orchestrator(dummy_request, [step1, step2])
    events: List[str] = [e async for e in orch.execute()]
    assert calls == ["step1", "step2"]
    assert "step1" in orch.context and "step2" in orch.context
    assert any("event1" in e for e in events)
    assert any("event2" in e for e in events)


@pytest.mark.asyncio
async def test_orchestrator_yields_correct_events(
    dummy_request: WorkflowRunRequest,
) -> None:
    emitter: EventEmitter = EventEmitter(conversation_id="conv-1")

    async def step(
        context: Dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str, None]:
        yield "step_event"

    with (
        patch.object(EventEmitter, "stream_start", return_value="start"),
        patch.object(
            EventEmitter,
            "workflow_step",
            side_effect=["step_started", "step_completed"],
        ),
        patch.object(EventEmitter, "stream_end", return_value="end"),
    ):
        orch: Orchestrator = Orchestrator(dummy_request, [step], context={})
        orch.emitter = emitter
        events: List[str] = [e async for e in orch.execute()]
        assert events[0] == "start"
        assert "step_started" in events
        assert "step_event" in events
        assert "step_completed" in events
        assert events[-1] == "end"


@pytest.mark.asyncio
async def test_orchestrator_handles_exceptions(
    dummy_request: WorkflowRunRequest,
) -> None:
    async def step_ok(
        context: Dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str, None]:
        yield "ok"

    async def step_fail(
        context: Dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str, None]:
        raise RuntimeError("fail!")
        yield  # pragma: no cover

    orch: Orchestrator = Orchestrator(dummy_request, [step_ok, step_fail])
    events: List[str] = [e async for e in orch.execute()]
    assert any("fail!" in e for e in events)
    assert not any("completed" in e for e in events if "fail!" in e)


@pytest.mark.asyncio
async def test_orchestrator_final_conversation_history(
    dummy_request: WorkflowRunRequest,
) -> None:
    async def step(
        context: Dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str, None]:
        context["final_response"] = "Assistant reply"
        yield "step"

    orch: Orchestrator = Orchestrator(dummy_request, [step])
    events: List[str] = [e async for e in orch.execute()]
    assert any("Assistant reply" in e for e in events)
    assert any("Hello?" in e for e in events)


@pytest.mark.asyncio
async def test_orchestrator_context_merging(dummy_request: WorkflowRunRequest) -> None:
    context: Dict[str, Any] = {"user_input": "should be overwritten", "foo": 42}

    async def step(
        context: Dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str, None]:
        yield "done"

    orch: Orchestrator = Orchestrator(dummy_request, [step], context=context)
    assert orch.context["user_input"] == dummy_request.user_input
    assert orch.context["foo"] == 42
