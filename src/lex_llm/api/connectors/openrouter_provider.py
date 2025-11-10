from typing import AsyncGenerator, List, Optional
import litellm
import os
from ..event_models import ConversationMessage
from .openai_provider import LLMProvider


class OpenRouterProvider(LLMProvider):
    """Implementation for OpenRouter's API with configurable model and provider."""

    def __init__(
        self,
        model: str = "google/gemma-3-27b-it",
        providers: Optional[List[str]] = ["nebius/fp8"],
    ):
        """
        Initialize OpenRouter provider.

        Args:
            model: The model to use (default: google/gemma-3-27b-it)
            provider: The provider preference for routing (default: nebius/fp8)
                     Set to None to use OpenRouter's default routing
            api_key: OpenRouter API key (default: reads from OPENROUTER_API_KEY env var)
        """
        self.model = model
        self.providers = providers

    async def generate_stream(
        self, messages: List[ConversationMessage]
    ) -> AsyncGenerator[str, None]:
        """Calls the OpenRouter API and streams the response."""
        # Prepare extra headers for provider routing
        extra_body = {}
        if self.providers:
            extra_body["provider"] = {"order": self.providers,
                                      "allow_fallbacks": False}

        stream = await litellm.acompletion(
            model=f"openrouter/{self.model}",
            messages=messages,
            stream=True,
            extra_body=extra_body if extra_body else None,
        )
        async for chunk in stream:  # type: ignore
            content = chunk.choices[0].delta.content
            if content:
                yield content

    async def generate(self, messages: List[ConversationMessage]) -> str:
        """Generates a response as a single text chunk."""
        response = ""
        async for chunk in self.generate_stream(messages):
            response += chunk
        return response
