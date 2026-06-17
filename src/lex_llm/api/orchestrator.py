import uuid
import asyncio
import time as time_module
from typing import Callable, Any, AsyncGenerator
from .event_emitter import EventEmitter
from .connectors.dgx_provider import set_run_id
from .event_models import (
    ConversationMessage,
    WorkflowRunRequest,
    WorkflowStepData,
    WorkflowMetricsData,
)
from .observability.run_recorder import get_recorder

StepFunc = Callable[
    [dict[str, Any], EventEmitter],
    AsyncGenerator[str | None, None],
]


class ParallelStep:
    """Wraps multiple step functions to be executed concurrently.

    All steps share the same context dict. Events yielded by each step
    are interleaved into the main event stream via an asyncio.Queue.
    If any step raises an exception, the others are cancelled and the
    error is propagated.
    """

    def __init__(self, steps: list[tuple[StepFunc, str]], label: str = "parallel"):
        self.steps = steps
        self.__name__ = label


class Orchestrator:
    def __init__(
        self,
        request: WorkflowRunRequest,
        steps: list[tuple[StepFunc | ParallelStep, str]],
        context: dict[str, Any] = {},
        workflow_id: str = "",
    ):
        self.request = request
        self.steps = steps
        self.workflow_id = workflow_id
        self.emitter = EventEmitter(conversation_id=request.conversation_id)
        # A simple dictionary to pass state between steps
        self.context: dict[str, Any] = {**context, **request.model_dump()}
        # Accumulated step telemetry for backend_summary aggregation
        self._step_telemetries: list[dict[str, Any]] = []

    async def _run_step(
        self, step_func: StepFunc, step_id: str, step_name: str, step_description: str | None
    ) -> AsyncGenerator[str, None]:
        """Run a single step, wrapping it with workflow_step events."""
        yield self.emitter.workflow_step(
            WorkflowStepData(
                step_id=step_id,
                name=step_name,
                status="started",
                description=step_description,
            )
        )

        # Allocate per-step telemetry so LLM steps can record backend info
        step_telemetry: dict[str, Any] = {}
        self.context["_current_step_telemetry"] = step_telemetry

        t_start = time_module.perf_counter()
        try:
            async for event in step_func(self.context, self.emitter):
                if event:
                    yield event
        except Exception as exc:
            duration_ms = (time_module.perf_counter() - t_start) * 1000
            self._step_telemetries.append(step_telemetry)
            yield self.emitter.workflow_step(
                WorkflowStepData(
                    step_id=step_id,
                    name=step_name,
                    status="failed",
                    description=step_description,
                    output={"duration_ms": duration_ms, **step_telemetry},
                    error=str(exc),
                )
            )
            raise
        finally:
            self.context.pop("_current_step_telemetry", None)

        duration_ms = (time_module.perf_counter() - t_start) * 1000
        self._step_telemetries.append(step_telemetry)
        yield self.emitter.workflow_step(
            WorkflowStepData(
                step_id=step_id,
                name=step_name,
                status="completed",
                description=step_description,
                output={"duration_ms": duration_ms, **step_telemetry},
            )
        )

    async def _run_parallel_step(
        self, parallel: ParallelStep
    ) -> AsyncGenerator[str, None]:
        """Run multiple steps concurrently, interleaving their events."""
        step_id = str(uuid.uuid4())
        step_name = parallel.__name__


        queue: asyncio.Queue[str | None] = asyncio.Queue()
        step_count = len(parallel.steps)

        async def _drain_step(step_func: StepFunc) -> None:
            """Run a step and push its events into the shared queue."""
            try:
                async for event in step_func(self.context, self.emitter):
                    if event:
                        await queue.put(event)
            except Exception as e:
                # Put the exception in the queue so the consumer can raise it
                await queue.put(e)  # type: ignore
            finally:
                await queue.put(None)  # Signal this step is done

        tasks = [asyncio.create_task(_drain_step(step)) for step, _ in parallel.steps]
        step_description = "Udfører følgende opgaver:\n\n" + "\n".join([description for _, description in parallel.steps])
        yield self.emitter.workflow_step(
            WorkflowStepData(
                step_id=step_id,
                name=step_name,
                status="started",
                description=step_description,
            )
        )
        try:
            completed = 0
            while completed < step_count:
                item = await queue.get()
                if item is None:
                    completed += 1
                elif isinstance(item, Exception):
                    raise item
                else:
                    yield item
        except Exception:
            # Cancel remaining tasks on error
            for task in tasks:
                task.cancel()
            raise
        finally:
            # Ensure all tasks are cleaned up
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Await tasks to suppress cancelled warnings
            await asyncio.gather(*tasks, return_exceptions=True)

        yield self.emitter.workflow_step(
            WorkflowStepData(
                step_id=step_id,
                name=step_name,
                status="completed",
                description=step_description,
            )
        )

    async def execute(self) -> AsyncGenerator[str, None]:
        """Executes the workflow steps and yields NDJSON events."""
        # Propagate run ID to DGXProvider for nginx trace correlation
        set_run_id(self.emitter.run_id)

        yield self.emitter.stream_start(
            conversation_history=self.request.conversation_history
        )

        t_start = time_module.perf_counter()
        was_deferral = False
        step_count = 0

        try:
            # Execute each step in the defined sequence
            for step, description in self.steps:
                if isinstance(step, ParallelStep):
                    async for p_event in self._run_parallel_step(step):
                        yield p_event
                else:
                    step_id = str(uuid.uuid4())
                    step_name = step.__name__
                    step_count += 1
                    async for event in self._run_step(step, step_id, step_name, description):
                        yield event
                # Check for early termination
                if self.context.get("_workflow_done"):
                    was_deferral = True
                    break

        except Exception as e:
            error_message = f"Workflow failed at step '{step_name}': {e}"  # type: ignore
            yield self.emitter.error(message=error_message)
            # Emit workflow_metrics even on error
            yield self._emit_workflow_metrics(t_start, step_count, "error")
            # Submit telemetry row
            await self._submit_recorder_row(t_start, "error")
            return  # Stop the generator

        # Determine outcome before building conversation history
        outcome: str = "deferral" if was_deferral else "ok"
        yield self._emit_workflow_metrics(t_start, step_count, outcome)
        await self._submit_recorder_row(t_start, outcome)

        # After all steps, construct the final history and end the stream
        final_assistant_message = self.context.get("final_response", "")
        user_message = self.context.get(
            "user_message_with_sources", self.request.user_input
        )
        system_prompt_with_sources = self.context.get("system_prompt", "")

        # Build the new history
        new_history = []
        if system_prompt_with_sources and not self.request.conversation_history:
            # First message: include system prompt with used sources
            new_history.append(
                ConversationMessage(role="system", content=system_prompt_with_sources)
            )

        new_history += [
            ConversationMessage(role="user", content=user_message),
            ConversationMessage(role="assistant", content=final_assistant_message),
        ]

        # For follow-up messages, update the system prompt in history
        if self.request.conversation_history and system_prompt_with_sources:
            # Replace old system message with updated one containing all used sources
            updated_history = [
                ConversationMessage(role="system", content=system_prompt_with_sources)
            ]
            # Add all user/assistant pairs from previous history
            for msg in self.request.conversation_history:
                if msg.role in ["user", "assistant"]:
                    updated_history.append(msg)
            # Add new user/assistant pair
            updated_history += new_history
        else:
            updated_history = self.request.conversation_history + new_history

        yield self.emitter.stream_end(conversation_history=updated_history)

    def _emit_workflow_metrics(
        self, t_start: float, step_count: int, outcome: str
    ) -> str:
        """Compute TTFT and e2e from internally recorded timestamps."""
        now = time_module.perf_counter()
        e2e_ms = (now - t_start) * 1000
        e = self.emitter
        ttft_any_ms = (
            (e._first_any_chunk_t - t_start) * 1000
            if e._first_any_chunk_t is not None
            else None
        )
        ttft_answer_ms = (
            (e._first_answer_chunk_t - t_start) * 1000
            if e._first_answer_chunk_t is not None
            else None
        )
        # Rolling backend summary from context steps that wrote to telemetry
        backend_counts = self._build_backend_summary()
        return e.workflow_metrics(
            WorkflowMetricsData(
                workflow_id=self.workflow_id,
                e2e_ms=e2e_ms,
                ttft_any_ms=ttft_any_ms,
                ttft_answer_ms=ttft_answer_ms,
                backend_summary=backend_counts,
                step_count=step_count,
                outcome=outcome,  # type: ignore[arg-type]
            )
        )

    def _build_backend_summary(self) -> dict[str, int]:
        """Aggregate backend counts across all completed step telemetries."""
        counts: dict[str, int] = {}
        for tel in self._step_telemetries:
            for call in tel.get("llm_calls") or []:
                b = call.get("backend", "unknown")
                counts[b] = counts.get(b, 0) + 1
        return counts

    async def _submit_recorder_row(self, t_start: float, outcome: str) -> None:
        """Submit one telemetry row to the JSONL recorder."""
        e = self.emitter
        now = time_module.perf_counter()
        row = {
            "ts": time_module.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "conversation_id": self.request.conversation_id,
            "run_id": e.run_id,
            "workflow_id": self.workflow_id,
            "outcome": outcome,
            "user_input_len": len(self.request.user_input),
            "e2e_ms": round((now - t_start) * 1000, 2),
            "ttft_any_ms": (
                round((e._first_any_chunk_t - t_start) * 1000, 2)
                if e._first_any_chunk_t is not None
                else None
            ),
            "ttft_answer_ms": (
                round((e._first_answer_chunk_t - t_start) * 1000, 2)
                if e._first_answer_chunk_t is not None
                else None
            ),
            "step_count": len(self.steps),
            "backend_summary": self._build_backend_summary(),
        }
        try:
            await get_recorder().submit(row)
        except Exception:
            pass
