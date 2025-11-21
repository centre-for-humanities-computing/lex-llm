import uuid
from typing import Callable, Dict, Any, AsyncGenerator, List
from .event_emitter import EventEmitter
from .event_models import ConversationMessage, WorkflowRunRequest, WorkflowStepData

StepFunc = Callable[
    [Dict[str, Any], EventEmitter],
    AsyncGenerator[str | None, None],
]


class Orchestrator:
    def __init__(
        self,
        request: WorkflowRunRequest,
        steps: List[StepFunc],
        context: Dict[str, Any] = {},
    ):
        self.request = request
        self.steps = steps
        self.emitter = EventEmitter(conversation_id=request.conversation_id)
        # A simple dictionary to pass state between steps
        self.context: Dict[str, Any] = {**context, **request.model_dump()}

    async def execute(self) -> AsyncGenerator[str, None]:
        """Executes the workflow steps and yields NDJSON events."""
        yield self.emitter.stream_start(
            conversation_history=self.request.conversation_history
        )

        try:
            # Execute each step in the defined sequence
            for step_func in self.steps:
                step_id = str(uuid.uuid4())
                step_name = step_func.__name__

                yield self.emitter.workflow_step(
                    WorkflowStepData(step_id=step_id, name=step_name, status="started")
                )

                # The actual execution of the step
                async for event in step_func(self.context, self.emitter):
                    if event:
                        yield event

                yield self.emitter.workflow_step(
                    WorkflowStepData(
                        step_id=step_id, name=step_name, status="completed"
                    )
                )

        except Exception as e:
            # Emit a detailed error and stop execution
            error_message = f"Workflow failed at step '{step_name}': {e}"  # type: ignore
            yield self.emitter.error(message=error_message)
            return  # Stop the generator

        # After all steps, construct the final history and end the stream
        final_assistant_message = self.context.get("final_response", "")
        system_prompt = self.context.get("system_prompt", "")
        user_message_with_sources = self.context.get("user_message_with_sources", self.request.user_input)
        
        # Build the new history
        new_history = []
        if system_prompt and not self.request.conversation_history:
            # Only add system prompt for the first message in a conversation
            new_history.append(
                ConversationMessage(role="system", content=system_prompt)
            )
        
        new_history += [
            ConversationMessage(role="user", content=user_message_with_sources),
            ConversationMessage(role="assistant", content=final_assistant_message),
        ]
        updated_history = self.request.conversation_history + new_history

        yield self.emitter.stream_end(conversation_history=updated_history)
