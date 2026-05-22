import uuid
import asyncio
from typing import Callable, Any, AsyncGenerator
from .event_emitter import EventEmitter
from .event_models import ConversationMessage, WorkflowRunRequest, WorkflowStepData

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

    def __init__(self, steps: list[StepFunc], label: str = "parallel"):
        self.steps = steps
        self.__name__ = label


class Orchestrator:
    def __init__(
        self,
        request: WorkflowRunRequest,
        steps: list[StepFunc | ParallelStep],
        context: dict[str, Any] = {},
    ):
        self.request = request
        self.steps = steps
        self.emitter = EventEmitter(conversation_id=request.conversation_id)
        # A simple dictionary to pass state between steps
        self.context: dict[str, Any] = {**context, **request.model_dump()}

    async def _run_step(
        self, step_func: StepFunc, step_id: str, step_name: str
    ) -> AsyncGenerator[str, None]:
        """Run a single step, wrapping it with workflow_step events."""
        yield self.emitter.workflow_step(
            WorkflowStepData(step_id=step_id, name=step_name, status="started")
        )

        async for event in step_func(self.context, self.emitter):
            if event:
                yield event

        yield self.emitter.workflow_step(
            WorkflowStepData(step_id=step_id, name=step_name, status="completed")
        )

    async def _run_parallel_step(
        self, parallel: ParallelStep
    ) -> AsyncGenerator[str, None]:
        """Run multiple steps concurrently, interleaving their events."""
        step_id = str(uuid.uuid4())
        step_name = parallel.__name__

        yield self.emitter.workflow_step(
            WorkflowStepData(step_id=step_id, name=step_name, status="started")
        )

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

        tasks = [asyncio.create_task(_drain_step(step)) for step in parallel.steps]

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
            WorkflowStepData(step_id=step_id, name=step_name, status="completed")
        )

    async def execute(self) -> AsyncGenerator[str, None]:
        """Executes the workflow steps and yields NDJSON events."""
        yield self.emitter.stream_start(
            conversation_history=self.request.conversation_history
        )

        try:
            # Execute each step in the defined sequence
            for step in self.steps:
                # Check for early termination
                if self.context.get("_workflow_done"):
                    break

                if isinstance(step, ParallelStep):
                    async for p_event in self._run_parallel_step(step):
                        yield p_event
                else:
                    step_id = str(uuid.uuid4())
                    step_name = step.__name__

                    async for event in self._run_step(step, step_id, step_name):
                        yield event

        except Exception as e:
            # Emit a detailed error and stop execution
            error_message = f"Workflow failed at step '{step_name}': {e}"  # type: ignore
            yield self.emitter.error(message=error_message)
            return  # Stop the generator

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
