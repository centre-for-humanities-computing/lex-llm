"""LLM provider for any OpenAI-compatible endpoint.

Unlike the provider-specific classes, nothing is read from the environment:
the caller supplies base_url, api_key, model and any provider-specific request
fields (``extra_body``), e.g. per request from the frontend.
"""

import os
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from openai import AsyncOpenAI

from .llm_provider import LLMProvider
from ..event_models import ConversationMessage
from ..observability.logging_setup import get_logger
from ...utils.errors import describe_exception

logger = get_logger()

# Same explicit timeout and retry policy as CortecsProvider: all calls stream,
# so the read timeout bounds the wait for the first token and every gap
# between tokens. Only a call that has not started streaming is retried.
_TIMEOUT = httpx.Timeout(float(os.getenv("GENERIC_LLM_TIMEOUT_S", "60")), connect=5.0)
_MAX_RETRIES = int(os.getenv("GENERIC_LLM_MAX_RETRIES", "1"))

# Provider error bodies can echo the request back, which would put user
# queries and retrieved article text into the log. Bound what we record.
_MAX_DETAIL = 500


def _detail(exc: Exception) -> str:
    """One-line, length-capped rendering of a provider exception and its cause."""
    text = describe_exception(exc)
    return text[:_MAX_DETAIL] + "..." if len(text) > _MAX_DETAIL else text


class GenericProvider(LLMProvider):
    """Implementation for any OpenAI-compatible chat completions API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the provider.

        Args:
            base_url: The OpenAI-compatible endpoint, e.g. https://api.cortecs.ai/v1.
            api_key: The API key for that endpoint.
            model: The model to use.
            extra_body: Provider-specific request fields, sent unchanged with
                        every call (e.g. Cortecs ``preference`` and
                        ``reasoning_effort``). None sends nothing extra.
        """
        self.base_url = base_url
        self.model = model
        self.extra_body = extra_body
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=_TIMEOUT,
            max_retries=_MAX_RETRIES,
        )

    async def generate_stream(
        self, messages: List[ConversationMessage]
    ) -> AsyncGenerator[str, None]:
        """Calls the endpoint and streams the response."""
        msg_dicts = [m.model_dump() for m in messages]
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=msg_dicts,  # type: ignore[arg-type]
                stream=True,
                extra_body=self.extra_body,
            )
        except Exception as exc:
            # Logging only -- re-raised unchanged. phase=open means the SDK
            # already retried _MAX_RETRIES times before giving up.
            logger.warning(
                "generic_llm_error phase=open base_url=%s model=%s type=%s detail=%s",
                self.base_url,
                self.model,
                type(exc).__name__,
                _detail(exc),
            )
            raise

        chunks = 0
        try:
            async for chunk in stream:  # type: ignore[union-attr]
                # Some providers send chunks with no choices (e.g. usage).
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if content:
                    chunks += 1
                    yield content
        except Exception as exc:
            # phase=stream means it failed mid-answer, which the SDK does not
            # retry at all.
            logger.warning(
                "generic_llm_error phase=stream base_url=%s model=%s type=%s "
                "chunks=%d detail=%s",
                self.base_url,
                self.model,
                type(exc).__name__,
                chunks,
                _detail(exc),
            )
            raise

        if chunks == 0:
            logger.warning(
                "generic_llm_empty_completion base_url=%s model=%s messages=%d",
                self.base_url,
                self.model,
                len(msg_dicts),
            )

    async def generate(self, messages: List[ConversationMessage]) -> str:
        """Generates a response as a single text chunk."""
        response = ""
        async for chunk in self.generate_stream(messages):
            response += chunk
        return response
