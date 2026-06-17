from typing import Any, Dict, List, Optional
import uuid
import time
from datetime import datetime, timezone
from .event_models import (
    Source,
    StreamEvent,
    WorkflowStepData,
    WorkflowMetricsData,
    ConversationMessage,
    StreamStartData,
    StreamEndData,
    ToolCallData,
    ErrorData,
    DefinitionItem,
    DefinitionsData,
)


class EventEmitter:
    def __init__(self, conversation_id: str, run_id: Optional[str] = None):
        self.conversation_id = conversation_id
        self.run_id = run_id or str(uuid.uuid4())
        # Monotonic timestamps for TTFT measurement
        self._first_any_chunk_t: float | None = None
        self._first_answer_chunk_t: float | None = None
        self._chunk_event_names = {
            "text_chunk",
            "lead_paragraph",
            "answer_body",
            "interpretation",
        }
        self._answer_event_names = {"text_chunk"}

    def _mark_first_chunk(self, event: str) -> None:
        now = time.monotonic()
        if event in self._chunk_event_names and self._first_any_chunk_t is None:
            self._first_any_chunk_t = now
        if event in self._answer_event_names and self._first_answer_chunk_t is None:
            self._first_answer_chunk_t = now

    def emit(self, event: str, data: Any = None) -> str:
        event_obj = StreamEvent(
            event=event,
            conversation_id=self.conversation_id,
            run_id=self.run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=data.model_dump(exclude_none=True)
            if hasattr(data, "model_dump")
            else data,
        )
        return event_obj.model_dump_json(exclude_none=True) + "\n"

    def stream_start(
        self, conversation_history: Optional[List[ConversationMessage]] = None
    ) -> str:
        """Emits the stream_start event using model_validate for type safety."""
        history_as_dicts = None
        if conversation_history:
            history_as_dicts = [msg.model_dump() for msg in conversation_history]

        # Use model_validate to create the object from a dictionary
        data = StreamStartData.model_validate(
            {"conversation_history": history_as_dicts}
        )
        return self.emit("stream_start", data)

    def text_chunk(self, text: str) -> str:
        self._mark_first_chunk("text_chunk")
        return self.emit("text_chunk", text)

    def reasoning_chunk(self, text: str) -> str:
        return self.emit("reasoning_chunk", text)

    def lead_paragraph_chunk(self, text: str) -> str:
        """Emits a lead paragraph text chunk."""
        self._mark_first_chunk("lead_paragraph")
        return self.emit("lead_paragraph", text)

    def answer_body_chunk(self, text: str) -> str:
        """Emits an answer body text chunk."""
        self._mark_first_chunk("answer_body")
        return self.emit("answer_body", text)

    def interpretation_chunk(self, text: str) -> str:
        """Emits an interpretation text chunk."""
        self._mark_first_chunk("interpretation")
        return self.emit("interpretation", text)

    def definitions(self, definitions: List[DefinitionItem]) -> str:
        """Emits a list of term definitions."""
        data = DefinitionsData(definitions=definitions)
        return self.emit("definitions", data)

    def workflow_metrics(self, data: WorkflowMetricsData) -> str:
        """Emits per-request workflow metrics (TTFT, e2e, backend summary)."""
        return self.emit("workflow_metrics", data)

    def tool_call(
        self,
        name: str,
        input_data: Dict[str, Any],
        description: Optional[str] = None,
    ) -> str:
        data = ToolCallData(name=name, input=input_data, description=description)
        return self.emit("tool_call", data)

    def tool_result(self, name: str, result_data: Dict[str, Any]) -> str:
        data = ToolCallData(name=name, input=result_data)
        return self.emit("tool_result", data)

    def sources(self, sources: List[Source]) -> str:
        return self.emit("sources", [s.model_dump() for s in sources])

    def workflow_step(self, step_data: WorkflowStepData) -> str:
        return self.emit("workflow_step", step_data)

    def stream_end(
        self, conversation_history: Optional[List[ConversationMessage]] = None
    ) -> str:
        """Emits the stream_end event using model_validate for type safety."""
        history_as_dicts = None
        if conversation_history:
            history_as_dicts = [msg.model_dump() for msg in conversation_history]

        # Use model_validate to create the object from a dictionary
        data = StreamEndData.model_validate({"conversation_history": history_as_dicts})
        return self.emit("stream_end", data)

    def error(self, message: str, code: Optional[str] = None) -> str:
        data = ErrorData(message=message, code=code)
        return self.emit("error", data)
