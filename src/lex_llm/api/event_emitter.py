from typing import Any, Dict, List, Optional
import uuid
from .event_models import (
    Source,
    StreamEvent,
    WorkflowStepData,
    ConversationMessage,
    StreamStartData,
    StreamEndData,
    ToolCallData,
    ErrorData,
)


class EventEmitter:
    def __init__(self, conversation_id: str, run_id: Optional[str] = None):
        self.conversation_id = conversation_id
        self.run_id = run_id or str(uuid.uuid4())

    def emit(self, event: str, data: Any = None) -> str:
        event_obj = StreamEvent(
            event=event,
            conversation_id=self.conversation_id,
            run_id=self.run_id,
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
        return self.emit("text_chunk", text)

    def reasoning_chunk(self, text: str) -> str:
        return self.emit("reasoning_chunk", text)

    def tool_call(self, name: str, input_data: Dict[str, Any]) -> str:
        data = ToolCallData(name=name, input=input_data)
        return self.emit("tool_call", data)

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
