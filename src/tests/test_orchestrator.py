from typing import AsyncGenerator, Any, Dict, List
import json
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


@pytest_asyncio.fixture
async def first_turn_request() -> WorkflowRunRequest:
    """A request with no prior conversation history (first turn)."""
    return WorkflowRunRequest(
        conversation_id="conv-2",
        user_input="Hvad er en fregat?",
        conversation_history=[],
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

    orch: Orchestrator = Orchestrator(dummy_request, [(step1, ""), (step2, "")])
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
        orch: Orchestrator = Orchestrator(dummy_request, [(step, "")], context={})
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

    orch: Orchestrator = Orchestrator(dummy_request, [(step_ok, ""), (step_fail, "")])
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

    orch: Orchestrator = Orchestrator(dummy_request, [(step, "")])
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

    orch: Orchestrator = Orchestrator(dummy_request, [(step, "")], context=context)
    assert orch.context["user_input"] == dummy_request.user_input
    assert orch.context["foo"] == 42


@pytest.mark.asyncio
async def test_first_turn_prepends_system_prompt(
    first_turn_request: WorkflowRunRequest,
) -> None:
    """First turn with system_prompt set should prepend a system message."""
    base_prompt = "Du er en chatbot."

    async def step(
        context: Dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str, None]:
        context["final_response"] = "Svar her."
        context["system_prompt"] = base_prompt
        yield "step"

    orch: Orchestrator = Orchestrator(
        first_turn_request, [(step, "")], use_clean_history=True
    )
    events: List[str] = [e async for e in orch.execute()]

    # Find the stream_end event and inspect its conversation_history
    end_event = next(e for e in events if "stream_end" in e)
    data = json.loads(end_event)
    history = data["data"]["conversation_history"]

    assert len(history) == 3  # system, user, assistant
    assert history[0]["role"] == "system"
    assert history[0]["content"] == base_prompt
    assert history[1]["role"] == "user"
    assert history[1]["content"] == first_turn_request.user_input
    assert history[2]["role"] == "assistant"
    assert history[2]["content"] == "Svar her."


@pytest.mark.asyncio
async def test_first_turn_no_system_prompt(
    first_turn_request: WorkflowRunRequest,
) -> None:
    """First turn without system_prompt should have no system message."""

    async def step(
        context: Dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str, None]:
        context["final_response"] = "Svar her."
        yield "step"

    orch: Orchestrator = Orchestrator(
        first_turn_request, [(step, "")], use_clean_history=True
    )
    events: List[str] = [e async for e in orch.execute()]

    end_event = next(e for e in events if "stream_end" in e)
    data = json.loads(end_event)
    history = data["data"]["conversation_history"]

    assert len(history) == 2  # user, assistant only
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_follow_up_preserves_existing_history(
    dummy_request: WorkflowRunRequest,
) -> None:
    """Follow-up turn preserves existing history as-is, appending new turn."""

    async def step(
        context: Dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str, None]:
        context["final_response"] = "Assistant reply"
        # Even if system_prompt is set, it should NOT rewrite the system message
        context["system_prompt"] = "Some new prompt"
        yield "step"

    orch: Orchestrator = Orchestrator(
        dummy_request, [(step, "")], use_clean_history=True
    )
    events: List[str] = [e async for e in orch.execute()]

    end_event = next(e for e in events if "stream_end" in e)
    data = json.loads(end_event)
    history = data["data"]["conversation_history"]

    # Original history had 1 user message; new turn adds user + assistant = 3 total
    assert len(history) == 3
    # The original user message is preserved
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hi!"
    # New user message
    assert history[1]["role"] == "user"
    assert history[1]["content"] == "Hello?"
    # New assistant message
    assert history[2]["role"] == "assistant"
    assert history[2]["content"] == "Assistant reply"


@pytest.mark.asyncio
async def test_follow_up_preserves_system_message(
    first_turn_request: WorkflowRunRequest,
) -> None:
    """Follow-up with a system message in history preserves it unchanged."""
    base_prompt = "Du er en chatbot."

    # Simulate a second turn: history has system + user + assistant from turn 1
    request = WorkflowRunRequest(
        conversation_id="conv-3",
        user_input="Fortæl mere.",
        conversation_history=[
            ConversationMessage(role="system", content=base_prompt),
            ConversationMessage(role="user", content="Hvad er en fregat?"),
            ConversationMessage(role="assistant", content="En fregat er..."),
        ],
    )

    async def step(
        context: Dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str, None]:
        context["final_response"] = "Mere info her."
        # Setting system_prompt should NOT overwrite the existing system message
        context["system_prompt"] = "Should be ignored"
        yield "step"

    orch: Orchestrator = Orchestrator(request, [(step, "")], use_clean_history=True)
    events: List[str] = [e async for e in orch.execute()]

    end_event = next(e for e in events if "stream_end" in e)
    data = json.loads(end_event)
    history = data["data"]["conversation_history"]

    assert len(history) == 5  # system + 2 user + 2 assistant
    assert history[0]["role"] == "system"
    assert history[0]["content"] == base_prompt  # unchanged
    assert history[1]["role"] == "user"
    assert history[1]["content"] == "Hvad er en fregat?"
    assert history[2]["role"] == "assistant"
    assert history[2]["content"] == "En fregat er..."
    assert history[3]["role"] == "user"
    assert history[3]["content"] == "Fortæl mere."
    assert history[4]["role"] == "assistant"
    assert history[4]["content"] == "Mere info her."
