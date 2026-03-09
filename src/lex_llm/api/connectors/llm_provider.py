from abc import ABC, abstractmethod
from typing import AsyncGenerator, List
from ..event_models import ConversationMessage


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
