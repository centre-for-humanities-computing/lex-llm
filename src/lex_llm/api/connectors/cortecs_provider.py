"""LLM provider for the Cortecs.ai inference API.

Cortecs exposes an OpenAI-compatible endpoint.  We use the ``openai``
Python SDK directly (not litellm) since Cortecs is not yet listed as a
litellm provider.
"""

import os
from typing import AsyncGenerator, List

from openai import AsyncOpenAI

from .llm_provider import LLMProvider
from ..event_models import ConversationMessage


class CortecsProvider(LLMProvider):
    """Implementation for Cortecs.ai's OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "gemma-4-26b-a4b-it",
        preference: str = "balanced",
    ):
        """Initialize Cortecs provider.

        Args:
            model: The model to use (default: gemma-4-26b-a4b-it).
            preference: Cortecs routing preference (default: "balanced").
                        Passed as ``preference`` in the request body per the
                        Cortecs quickstart guide.
        """
        self.model = model
        self.preference = preference
        self._client = AsyncOpenAI(
            api_key=os.getenv("CORTECS_API_KEY", ""),
            base_url=os.getenv("CORTECS_BASE_URL", "https://api.cortecs.ai/v1"),
        )

    async def generate_stream(
        self, messages: List[ConversationMessage]
    ) -> AsyncGenerator[str, None]:
        """Calls the Cortecs API and streams the response."""
        msg_dicts = [m.model_dump() for m in messages]
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=msg_dicts,  # type: ignore[arg-type]
            stream=True,
            extra_body={"preference": self.preference},
        )
        async for chunk in stream:  # type: ignore[union-attr]
            content = chunk.choices[0].delta.content
            if content:
                yield content

    async def generate(self, messages: List[ConversationMessage]) -> str:
        """Generates a response as a single text chunk."""
        response = ""
        async for chunk in self.generate_stream(messages):
            response += chunk
        return response
