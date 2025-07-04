from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union


# Request/Response Models
class ConversationMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class ConversationHistoryData(BaseModel):
    conversation_history: List[ConversationMessage]


class WorkflowRunRequest(BaseModel):
    user_input: str
    conversation_history: List[ConversationMessage] = Field(default_factory=list)
    conversation_id: str = Field(
        ..., description="A unique identifier for the conversation (UUID)"
    )
    # session_id removed, not in docs


class Source(BaseModel):
    id: Union[str, int]
    title: str
    url: str


# Event Data Models
class StreamStartData(BaseModel):
    conversation_history: Optional[List[ConversationMessage]] = None


class StreamEndData(BaseModel):
    conversation_history: Optional[List[ConversationMessage]] = None


class ToolCallData(BaseModel):
    name: str
    input: Dict[str, Any]


class WorkflowStepData(BaseModel):
    step_id: str
    name: str
    status: str = Field(..., pattern="^(started|in_progress|completed|failed)$")
    input: Optional[Any] = None
    update: Optional[Any] = None
    output: Optional[Any] = None
    error: Optional[str] = None


class ErrorData(BaseModel):
    code: Optional[str] = None
    message: str


class StreamEvent(BaseModel):
    event: str
    conversation_id: str
    run_id: str
    data: Optional[Any] = None
