import litellm
from ..event_models import ConversationMessage
from .llm_provider import LLMProvider
import os
from typing import AsyncGenerator, List


class ScalewayProvider(LLMProvider):
    """Implementation for OpenAI's API."""

    def __init__(self, model: str = "gemma-3-27b-it"):
        os.environ["OPENAI_API_KEY"] = os.environ["SCW_SECRET_KEY"]
        os.environ["OPENAI_BASE_URL"] = (
            "https://api.scaleway.ai/" + os.environ["SCALEWAY_ORGID"] + "/v1"
        )
        self.model = model

    async def generate_stream(
        self, messages: List[ConversationMessage]
    ) -> AsyncGenerator[str, None]:
        """Calls the Scaleways API and streams the response."""
        stream = await litellm.acompletion(
            model="openai/" + self.model, messages=messages, stream=True
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
