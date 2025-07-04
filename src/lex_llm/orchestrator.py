import json
from typing import AsyncGenerator, Callable, List
from lex_llm.api.event_emitter import EventEmitter
from lex_llm.api.event_models import (
    WorkflowRunRequest,
    ConversationMessage,
)

# This file now only contains the orchestrator class. Workflow logic is in workflows/test_workflow.py


class WorkflowOrchestrator:
    def __init__(
        self,
        workflow_id: str,
        workflow_func: Callable[[WorkflowRunRequest], AsyncGenerator[str, None]],
    ) -> None:
        self.workflow_id: str = workflow_id
        self.workflow_func: Callable[
            [WorkflowRunRequest], AsyncGenerator[str, None]
        ] = workflow_func
        self.conversation_history: List[ConversationMessage] = []
        self._assistant_chunks: List[str] = []

    async def execute(self, request: WorkflowRunRequest) -> AsyncGenerator[str, None]:
        """Execute workflow and yield NDJSON events, handling start/stop and conversation history."""
        emitter = EventEmitter(conversation_id=request.conversation_id)
        self.conversation_history = request.conversation_history.copy()
        # Add user input to conversation history
        self.conversation_history.append(
            ConversationMessage(role="user", content=request.user_input)
        )
        self._assistant_chunks = []
        # Emit stream_start event
        yield emitter.stream_start(conversation_history=self.conversation_history)
        try:
            async for event in self._event_stream_with_history(request):
                yield event
            # On stream end, add assistant message to conversation history
            if self._assistant_chunks:
                full_assistant_message = "".join(self._assistant_chunks)
                self.conversation_history.append(
                    ConversationMessage(
                        role="assistant", content=full_assistant_message
                    )
                )
            # Emit stream_end event with final conversation history
            yield emitter.stream_end(conversation_history=self.conversation_history)
        except Exception as e:
            yield emitter.error(str(e))

    async def _event_stream_with_history(
        self, request: WorkflowRunRequest
    ) -> AsyncGenerator[str, None]:
        """Run the workflow and update conversation history by inspecting events."""

        async for event in self.workflow_func(request):
            # Try to parse the event as JSON to inspect its type
            try:
                event_obj = json.loads(event)
                # Accumulate assistant message chunks
                if event_obj.get("event") in ("text_chunk", "reasoning_chunk"):
                    chunk = event_obj.get("data")
                    if chunk:
                        self._assistant_chunks.append(chunk)
                elif event_obj.get("event") == "sources":
                    sources = event_obj.get("data", [])
                    # Convert sources to a markdown list format
                    sources_md = (
                        "\n\n## Sources:\n"
                        + "\n".join([f"- [{s['title']}]({s['url']})" for s in sources])
                        + "\n\n"
                    )

                    self._assistant_chunks.append(sources_md)
            except Exception:
                pass  # If not JSON, just yield
            yield event
