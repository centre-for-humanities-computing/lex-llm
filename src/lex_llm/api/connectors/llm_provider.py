from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, List, Literal
from ..event_models import ConversationMessage


@dataclass
class RouteDecision:
    """Captures the routing decision made by a RoutingLLMProvider.

    Attributes:
        backend: Which backend was selected.
        trigger: The reason category for the decision.
        reason: Human-readable explanation from the probe or exception.
        model: The model name on the selected backend.
    """

    backend: Literal["primary", "fallback"]
    trigger: Literal[
        "ok",
        "probe_overload",
        "probe_scrape_error",
        "primary_pre_first_token_error",
    ]
    reason: str = ""
    model: str = ""


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    async def generate_stream(
        self, messages: List[ConversationMessage]
    ) -> AsyncGenerator[str, None]:
        """Generates a response as a stream of text chunks."""
        yield ""

    @abstractmethod
    async def generate(self, messages: List[ConversationMessage]) -> str:
        """Generates a response as a single text chunk."""
        return ""

    @asynccontextmanager
    async def observe(
        self,
        callback: "Callable[[RouteDecision], None] | None" = None,
        *,
        telemetry: "dict[str, Any] | None" = None,
    ) -> "AsyncGenerator[None, None]":
        """No-op context manager; non-routing providers don't observe.

        Override in RoutingLLMProvider to capture routing decisions.
        """
        yield
