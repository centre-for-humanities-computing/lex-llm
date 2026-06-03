"""Tests for workflow observability (TTFT, backend routing, step timing, RunRecorder)."""

import asyncio
import json
import os
import tempfile
from typing import AsyncGenerator, Any, List
import pytest
import pytest_asyncio

from lex_llm.api.orchestrator import Orchestrator
from lex_llm.api.event_emitter import EventEmitter
from lex_llm.api.event_models import WorkflowRunRequest, ConversationMessage
from lex_llm.api.connectors.llm_provider import LLMProvider, RouteDecision
from lex_llm.api.connectors.vllm_load_probe import VLLMLoadProbe
from lex_llm.api.observability.run_recorder import RunRecorder


# ── Fake LLM providers ───────────────────────────────────────────────


class FakeLLM(LLMProvider):
    """LLM provider that yields fixed chunks after a configurable delay."""

    def __init__(self, model: str = "fake-model", delay: float = 0.001):
        self.model = model
        self.delay = delay

    async def generate_stream(
        self, messages: List[ConversationMessage]
    ) -> AsyncGenerator[str, None]:
        await asyncio.sleep(self.delay)
        for token in ["Hello", " world", "!"]:
            await asyncio.sleep(self.delay)
            yield token

    async def generate(self, messages: List[ConversationMessage]) -> str:
        out = ""
        async for chunk in self.generate_stream(messages):
            out += chunk
        return out


class FailingPrimaryLLM(LLMProvider):
    """LLM that raises before yielding a first token."""

    def __init__(self, model: str = "failing-model"):
        self.model = model

    async def generate_stream(
        self, messages: List[ConversationMessage]
    ) -> AsyncGenerator[str, None]:
        raise RuntimeError("primary is on fire")

    async def generate(self, messages: List[ConversationMessage]) -> str:
        raise RuntimeError("primary is on fire")


class EmptyPrimaryLLM(LLMProvider):
    """LLM that yields nothing (empty response)."""

    def __init__(self, model: str = "empty-model"):
        self.model = model

    async def generate_stream(
        self, messages: List[ConversationMessage]
    ) -> AsyncGenerator[str, None]:
        return
        yield  # type: ignore  # pragma: no cover

    async def generate(self, messages: List[ConversationMessage]) -> str:
        return ""


# ── Fake VLLM probe ──────────────────────────────────────────────────


class FakeProbe(VLLMLoadProbe):
    """Probe that returns a pre-configured overload state."""

    def __init__(self, overloaded: bool, reason: str = "ok"):
        super().__init__(
            metrics_url="http://fake/metrics",
            model_name="fake-model",
        )
        self._overloaded = overloaded
        self._reason = reason

    async def is_overloaded(self) -> tuple[bool, str]:
        return self._overloaded, self._reason

    async def _scrape(self) -> tuple[bool, str]:
        return self._overloaded, self._reason


# ── Step helpers ─────────────────────────────────────────────────────


async def _noop_step(
    context: dict[str, Any], emitter: EventEmitter
) -> AsyncGenerator[str | None, None]:
    yield None


async def _streaming_step(
    context: dict[str, Any], emitter: EventEmitter
) -> AsyncGenerator[str | None, None]:
    """Step that emits a text_chunk so TTFT gets measured."""
    yield emitter.text_chunk("test output")


async def _ddg_step(
    context: dict[str, Any], emitter: EventEmitter
) -> AsyncGenerator[str | None, None]:
    """Step that sets _workflow_done (simulates deferral)."""
    context["_workflow_done"] = True
    yield None


async def _failing_step(
    context: dict[str, Any], emitter: EventEmitter
) -> AsyncGenerator[str | None, None]:
    """Step that raises."""
    raise ValueError("step exploded")
    yield  # type: ignore  # pragma: no cover


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def request_fixture() -> WorkflowRunRequest:
    return WorkflowRunRequest(
        conversation_id="obs-test-1",
        user_input="test query",
        conversation_history=[],
    )


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_workflow_metrics_event_emitted(
    request_fixture: WorkflowRunRequest,
) -> None:
    """After a normal run the workflow_metrics event must precede stream_end."""
    orch = Orchestrator(request_fixture, [_noop_step], workflow_id="test_wf")
    events = [e async for e in orch.execute()]

    # Find workflow_metrics and stream_end
    metrics_idx = None
    end_idx = None
    for i, ev in enumerate(events):
        if '"workflow_metrics"' in ev:
            metrics_idx = i
        if '"stream_end"' in ev:
            end_idx = i

    assert metrics_idx is not None, "workflow_metrics event not found"
    assert end_idx is not None, "stream_end event not found"
    # workflow_metrics must come before stream_end
    assert metrics_idx < end_idx, "workflow_metrics must precede stream_end"

    # Parse and validate shape
    parsed = json.loads(events[metrics_idx].strip())
    assert parsed["event"] == "workflow_metrics"
    data = parsed["data"]
    assert data["workflow_id"] == "test_wf"
    assert data["e2e_ms"] > 0
    assert data["step_count"] >= 0
    assert data["outcome"] in ("ok", "error", "deferral")


@pytest.mark.asyncio
async def test_ttft_measured_on_text_chunk(
    request_fixture: WorkflowRunRequest,
) -> None:
    """When a step emits text_chunk, ttft_answer_ms and ttft_any_ms must be set."""
    orch = Orchestrator(request_fixture, [_streaming_step], workflow_id="ttft_test")
    events = [e async for e in orch.execute()]

    metrics = None
    for ev in events:
        if '"workflow_metrics"' in ev:
            metrics = json.loads(ev.strip())["data"]
            break

    assert metrics is not None
    assert metrics["ttft_any_ms"] is not None, "ttft_any_ms must be set"
    assert metrics["ttft_any_ms"] > 0, "ttft_any_ms must be > 0"
    assert metrics["ttft_answer_ms"] is not None, "ttft_answer_ms must be set"
    assert metrics["ttft_answer_ms"] > 0, "ttft_answer_ms must be > 0"


@pytest.mark.asyncio
async def test_deferral_outcome(
    request_fixture: WorkflowRunRequest,
) -> None:
    """When a step sets _workflow_done, outcome must be 'deferral'."""
    orch = Orchestrator(
        request_fixture, [_ddg_step], workflow_id="defer_test"
    )
    events = [e async for e in orch.execute()]

    metrics = None
    for ev in events:
        if '"workflow_metrics"' in ev:
            metrics = json.loads(ev.strip())["data"]
            break

    assert metrics is not None
    assert metrics["outcome"] == "deferral", (
        f"Expected deferral, got {metrics['outcome']}"
    )


@pytest.mark.asyncio
async def test_error_outcome(
    request_fixture: WorkflowRunRequest,
) -> None:
    """When a step raises, outcome must be 'error' and workflow_metrics still emitted."""
    orch = Orchestrator(
        request_fixture, [_failing_step], workflow_id="err_test"
    )
    events = [e async for e in orch.execute()]

    # Should have an error event
    assert any('"error"' in e for e in events), "error event must be emitted"

    metrics = None
    for ev in events:
        if '"workflow_metrics"' in ev:
            metrics = json.loads(ev.strip())["data"]
            break

    assert metrics is not None, "workflow_metrics must be emitted even on error"
    assert metrics["outcome"] == "error"


@pytest.mark.asyncio
async def test_step_duration_in_output(
    request_fixture: WorkflowRunRequest,
) -> None:
    """Each completed workflow_step event must carry duration_ms in output."""
    orch = Orchestrator(request_fixture, [_noop_step], workflow_id="dur_test")
    events = [e async for e in orch.execute()]

    completed_steps = [
        json.loads(e.strip())
        for e in events
        if '"workflow_step"' in e and '"completed"' in e
    ]
    assert len(completed_steps) >= 1
    for step in completed_steps:
        output = step["data"].get("output", {})
        assert output.get("duration_ms") is not None, (
            f"duration_ms missing in step {step['data']['name']}"
        )
        assert output["duration_ms"] >= 0, "duration_ms must be >= 0"


@pytest.mark.asyncio
async def test_route_callback_captured_in_step_telemetry(
    request_fixture: WorkflowRunRequest,
) -> None:
    """When using RoutingLLMProvider, llm_calls must appear in step telemetry."""
    from lex_llm.api.connectors.routing_llm_provider import RoutingLLMProvider

    primary = FakeLLM(model="primary-model")
    fallback = FakeLLM(model="fallback-model")
    probe = FakeProbe(overloaded=False)
    provider = RoutingLLMProvider(primary, fallback, probe)

    async def _llm_step(
        ctx: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str | None, None]:
        telemetry = ctx.setdefault("_current_step_telemetry", {})

        def _record(d: RouteDecision) -> None:
            telemetry.setdefault("llm_calls", []).append(
                {
                    "backend": d.backend,
                    "trigger": d.trigger,
                    "reason": d.reason,
                    "model": d.model,
                }
            )

        async with provider.observe(_record):
            result = await provider.generate(
                [ConversationMessage(role="user", content="hello")]
            )
        yield emitter.text_chunk(result)

    orch = Orchestrator(request_fixture, [_llm_step], workflow_id="route_test")
    events = [e async for e in orch.execute()]

    # Find the completed step output
    completed = [
        json.loads(e.strip())["data"]
        for e in events
        if '"workflow_step"' in e and '"completed"' in e
    ]
    assert len(completed) >= 1
    output = completed[0].get("output", {})
    calls = output.get("llm_calls", [])
    assert len(calls) == 1, f"Expected 1 llm_call, got {len(calls)}"
    assert calls[0]["backend"] == "primary"
    assert calls[0]["model"] == "primary-model"


@pytest.mark.asyncio
async def test_fallback_on_overload(
    request_fixture: WorkflowRunRequest,
) -> None:
    """When probe says overloaded, routing must go to fallback."""
    from lex_llm.api.connectors.routing_llm_provider import RoutingLLMProvider

    primary = FakeLLM(model="primary-model")
    fallback = FakeLLM(model="fallback-model")
    probe = FakeProbe(overloaded=True, reason="queue depth 5 >= 4")
    provider = RoutingLLMProvider(primary, fallback, probe)

    async def _llm_step(
        ctx: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str | None, None]:
        telemetry = ctx.setdefault("_current_step_telemetry", {})

        def _record(d: RouteDecision) -> None:
            telemetry.setdefault("llm_calls", []).append(
                {
                    "backend": d.backend,
                    "trigger": d.trigger,
                    "reason": d.reason,
                    "model": d.model,
                }
            )

        async with provider.observe(_record):
            result = await provider.generate(
                [ConversationMessage(role="user", content="hello")]
            )
        yield emitter.text_chunk(result)

    orch = Orchestrator(request_fixture, [_llm_step], workflow_id="fallback_test")
    events = [e async for e in orch.execute()]

    completed = [
        json.loads(e.strip())["data"]
        for e in events
        if '"workflow_step"' in e and '"completed"' in e
    ]
    assert len(completed) >= 1
    output = completed[0].get("output", {})
    calls = output.get("llm_calls", [])
    assert len(calls) == 1
    assert calls[0]["backend"] == "fallback"
    assert calls[0]["trigger"] == "probe_overload"


@pytest.mark.asyncio
async def test_fallback_on_primary_error(
    request_fixture: WorkflowRunRequest,
) -> None:
    """When primary raises before first token, routing must go to fallback."""
    from lex_llm.api.connectors.routing_llm_provider import RoutingLLMProvider

    primary = FailingPrimaryLLM(model="failing")
    fallback = FakeLLM(model="fallback-model")
    probe = FakeProbe(overloaded=False)
    provider = RoutingLLMProvider(primary, fallback, probe)

    async def _llm_step(
        ctx: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str | None, None]:
        telemetry = ctx.setdefault("_current_step_telemetry", {})

        def _record(d: RouteDecision) -> None:
            telemetry.setdefault("llm_calls", []).append(
                {
                    "backend": d.backend,
                    "trigger": d.trigger,
                    "reason": d.reason,
                    "model": d.model,
                }
            )

        async with provider.observe(_record):
            result = await provider.generate(
                [ConversationMessage(role="user", content="hello")]
            )
        yield emitter.text_chunk(result)

    orch = Orchestrator(request_fixture, [_llm_step], workflow_id="failover_test")
    events = [e async for e in orch.execute()]

    completed = [
        json.loads(e.strip())["data"]
        for e in events
        if '"workflow_step"' in e and '"completed"' in e
    ]
    assert len(completed) >= 1
    output = completed[0].get("output", {})
    calls = output.get("llm_calls", [])
    assert len(calls) == 1
    assert calls[0]["backend"] == "fallback"
    assert calls[0]["trigger"] == "primary_pre_first_token_error"


@pytest.mark.asyncio
async def test_workflow_metrics_e2e_positive(
    request_fixture: WorkflowRunRequest,
) -> None:
    """e2e_ms should be a positive number for any non-empty workflow."""
    orch = Orchestrator(request_fixture, [_noop_step], workflow_id="e2e_test")
    events = [e async for e in orch.execute()]

    metrics = None
    for ev in events:
        if '"workflow_metrics"' in ev:
            metrics = json.loads(ev.strip())["data"]
            break

    assert metrics is not None
    assert metrics["e2e_ms"] > 0, "e2e_ms must be positive"


@pytest.mark.asyncio
async def test_run_recorder_writes_row() -> None:
    """RunRecorder must write a parseable JSONL row."""
    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = RunRecorder(directory=tmpdir, max_queue=100)
        await recorder.start()

        row = {
            "ts": "2026-06-03T12:00:00+0000",
            "run_id": "rec-test-1",
            "workflow_id": "rec_test_wf",
            "outcome": "ok",
            "e2e_ms": 123.45,
            "ttft_any_ms": 50.0,
        }
        await recorder.submit(row)
        await recorder.stop()

        # Check file was written
        files = os.listdir(tmpdir)
        assert len(files) == 1
        assert files[0].endswith(".jsonl")

        with open(os.path.join(tmpdir, files[0])) as f:
            line = f.readline().strip()
        parsed = json.loads(line)
        assert parsed["run_id"] == "rec-test-1"
        assert parsed["e2e_ms"] == 123.45