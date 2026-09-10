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
from ..observability.logging_setup import get_logger

logger = get_logger()

# Provider error bodies can echo the request back, which would put user
# queries and retrieved article text into the log. Bound what we record.
_MAX_DETAIL = 500


def _detail(exc: Exception) -> str:
    """One-line, length-capped rendering of a provider exception."""
    text = str(exc).replace("\n", " ")
    return text[:_MAX_DETAIL] + "..." if len(text) > _MAX_DETAIL else text


class CortecsProvider(LLMProvider):
    """Implementation for Cortecs.ai's OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "gemma-4-26b-a4b-it",
        preference: str = "balanced",
        reasoning_effort: str = "low",
    ):
        """Initialize Cortecs provider.

        Args:
            model: The model to use (default: gemma-4-26b-a4b-it).
            preference: Cortecs routing preference (default: "balanced").
                        Passed as ``preference`` in the request body per the
                        Cortecs quickstart guide.
            reasoning_effort: The reasoning effort level (default: "low").
                                Passed as ``reasoning_effort`` in the request body.
        """
        self.model = model
        self.preference = preference
        self.reasoning_effort = reasoning_effort
        self._client = AsyncOpenAI(
            api_key=os.getenv("CORTECS_API_KEY", ""),
            base_url=os.getenv("CORTECS_BASE_URL", "https://api.cortecs.ai/v1"),
        )

    async def generate_stream(
        self, messages: List[ConversationMessage]
    ) -> AsyncGenerator[str, None]:
        """Calls the Cortecs API and streams the response."""
        msg_dicts = [m.model_dump() for m in messages]
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=msg_dicts,  # type: ignore[arg-type]
                stream=True,
                extra_body={
                    "preference": self.preference,
                    "reasoning_effort": self.reasoning_effort,
                },
            )
        except Exception as exc:
            # Logging only -- re-raised unchanged. phase=open means the SDK
            # already retried twice before giving up.
            logger.warning(
                "cortecs_error phase=open model=%s type=%s detail=%s",
                self.model,
                type(exc).__name__,
                _detail(exc),
            )
            raise

        chunks = 0
        try:
            async for chunk in stream:  # type: ignore[union-attr]
                content = chunk.choices[0].delta.content
                if content:
                    chunks += 1
                    yield content
        except Exception as exc:
            # phase=stream means it failed mid-answer, which the SDK does not
            # retry at all.
            logger.warning(
                "cortecs_error phase=stream model=%s type=%s chunks=%d detail=%s",
                self.model,
                type(exc).__name__,
                chunks,
                _detail(exc),
            )
            raise

        if chunks == 0:
            logger.warning(
                "cortecs_empty_completion model=%s messages=%d",
                self.model,
                len(msg_dicts),
            )

    async def generate(self, messages: List[ConversationMessage]) -> str:
        """Generates a response as a single text chunk."""
        response = ""
        async for chunk in self.generate_stream(messages):
            response += chunk
        return response
