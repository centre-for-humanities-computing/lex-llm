from pydantic import BaseModel, Field, SecretStr, field_validator
from typing import Dict, List, Optional, Any, Union, Literal


# Request/Response Models
class ConversationMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer", "tool", "function"]
    content: str


class LLMConfig(BaseModel):
    """LLM endpoint chosen by the caller, for workflows that accept one."""

    base_url: str = Field(..., min_length=1)
    api_key: SecretStr = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    extra_body: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Provider-specific request fields, sent unchanged",
    )

    @field_validator("base_url", "model", "api_key")
    @classmethod
    def _not_blank(cls, v: Union[str, SecretStr]) -> Union[str, SecretStr]:
        raw = v.get_secret_value() if isinstance(v, SecretStr) else v
        if not raw.strip():
            raise ValueError("must not be blank")
        return v

    @field_validator("extra_body")
    @classmethod
    def _no_reserved_keys(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        # The SDK merges extra_body over the request body, so these would
        # silently replace the fields the provider sets itself.
        reserved = {"model", "messages", "stream"} & set(v or {})
        if reserved:
            raise ValueError(
                f"must not contain {sorted(reserved)}; set the model in 'model'"
            )
        return v


class WorkflowRunRequest(BaseModel):
    user_input: str
    conversation_history: List[ConversationMessage] = Field(default_factory=list)
    conversation_id: str = Field(
        ..., description="A unique identifier for the conversation (UUID)"
    )
    llm: Optional[LLMConfig] = None


class Source(BaseModel):
    id: Union[str, int]
    title: str
    url: str | None = None
    highlight: str | None = None


class DefinitionItem(BaseModel):
    term: str
    definition: str


# Event Data Models
class StreamStartData(BaseModel):
    conversation_history: Optional[List[ConversationMessage]] = None


class StreamEndData(BaseModel):
    conversation_history: Optional[List[ConversationMessage]] = None


class ToolCallData(BaseModel):
    name: str
    input: Dict[str, Any]
    description: Optional[str] = None


class WorkflowStepData(BaseModel):
    step_id: str
    name: str
    status: str = Field(..., pattern="^(started|in_progress|completed|failed)$")
    description: Optional[str] = None
    input: Optional[Any] = None
    update: Optional[Any] = None
    output: Optional[Any] = None
    error: Optional[str] = None


class ErrorData(BaseModel):
    code: Optional[str] = None
    message: str


class DefinitionsData(BaseModel):
    definitions: List[DefinitionItem]


class WorkflowMetricsData(BaseModel):
    """Emitted right before stream_end with per-request telemetry."""

    workflow_id: str
    e2e_ms: float
    ttft_any_ms: float | None = None
    ttft_answer_ms: float | None = None
    backend_summary: Dict[str, int] = Field(default_factory=dict)
    step_count: int = 0
    outcome: Literal["ok", "error", "deferral"] = "ok"


class StreamEvent(BaseModel):
    event: str
    conversation_id: str
    run_id: str
    timestamp: Optional[str] = None
    data: Optional[Any] = None
