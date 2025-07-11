from abc import ABC, abstractmethod
from typing import AsyncGenerator, List
import litellm
from ..event_models import ConversationMessage


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    async def generate_stream(
        self, messages: List[ConversationMessage]
    ) -> AsyncGenerator[str, None]:
        """Generates a response as a stream of text chunks."""
        yield ""


class OpenAIProvider(LLMProvider):
    """Implementation for OpenAI's API."""

    async def generate_stream(
        self, messages: List[ConversationMessage]
    ) -> AsyncGenerator[str, None]:
        """Calls the OpenAI API and streams the response."""
        stream = await litellm.acompletion(
            model="gpt-4.1", messages=messages, stream=True
        )
        async for chunk in stream:  # type: ignore
            content = chunk.choices[0].delta.content
            if content:
                yield content
