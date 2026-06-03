# llm/dgx_provider.py
import logging
import os
from typing import AsyncGenerator, List
import litellm
from ..event_models import ConversationMessage
from .llm_provider import LLMProvider

logger = logging.getLogger(__name__)

class DGXProvider(LLMProvider):
    """Talks to the DGX Spark via the nginx-fronted vLLM OpenAI endpoint.

    The nginx server routes by path prefix using the model name itself.
    ``INFERENCE_SERVER_URL`` should be the bare host, e.g.
    ``http://10.57.5.14:80``.  The provider constructs the full
    ``api_base`` as ``{server_url}/{model}/v1`` so the model name
    in the URL matches the ``id`` returned by ``/v1/models``.
    """

    def __init__(
        self,
        model: str = "gemma-4-26B-A4B-it",
        server_url: str | None = None,
    ):
        self.model = model
        _server = (server_url or os.environ["INFERENCE_SERVER_URL"]).rstrip("/")
        self.base_url = f"{_server}/{model}/v1"

    async def generate_stream(
        self, messages: List[ConversationMessage]
    ) -> AsyncGenerator[str, None]:
        messages_dicts = [m.model_dump() for m in messages]
        stream = await litellm.acompletion(
            model=self.model,
            messages=messages_dicts,
            stream=True,
            api_base=self.base_url,
            api_key="not-needed",
            timeout=30,
            custom_llm_provider="openai",
        )
        async for chunk in stream:  # type: ignore
            content = chunk.choices[0].delta.content
            if content:
                yield content

    async def generate(self, messages: List[ConversationMessage]) -> str:
        out = ""
        async for chunk in self.generate_stream(messages):
            out += chunk
        return out
