# llm/dgx_provider.py
import contextvars
import logging
import os
from typing import AsyncGenerator, List
import litellm
from ..event_models import ConversationMessage
from .llm_provider import LLMProvider

logger = logging.getLogger(__name__)

# Context variable for trace ID propagation to the inference server
_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_dgx_run_id", default=None
)


def set_run_id(run_id: str | None) -> None:
    """Set the run ID for the current async task.

    Called by the orchestrator before invoking steps so that DGXProvider
    can propagate the ID as an ``X-Lex-Run-Id`` header to nginx access logs.
    """
    if run_id:
        _run_id.set(run_id)


class DGXProvider(LLMProvider):
    """Talks to the DGX Spark via the nginx-fronted vLLM OpenAI endpoint.

    The nginx server routes by path prefix using the model name itself.
    ``INFERENCE_SERVER_URL`` should be the bare host, e.g.
    ``http://10.57.5.14:80``.  The provider constructs the full
    ``api_base`` as ``{server_url}/{model}/v1`` so the model name
    in the URL matches the ``id`` returned by ``/v1/models``.

    Raises ``RuntimeError`` from ``generate_stream`` if
    ``INFERENCE_SERVER_XAUTH`` is not set, so that routing providers
    can detect the misconfiguration and fall back to an alternative.
    """

    def __init__(
        self,
        model: str = "gemma-4-26B-A4B-it",
        server_url: str | None = None,
    ):
        self.model = model
        _server = (server_url or os.environ["INFERENCE_SERVER_URL"]).rstrip("/")
        self.base_url = f"{_server}/{model}/v1"
        self._xauth_token: str | None = os.environ.get("INFERENCE_SERVER_XAUTH")

    async def generate_stream(
        self, messages: List[ConversationMessage]
    ) -> AsyncGenerator[str, None]:
        if not self._xauth_token:
            raise RuntimeError(
                "INFERENCE_SERVER_XAUTH is not set; cannot call DGX inference server"
            )
        messages_dicts = [m.model_dump() for m in messages]
        run_id = _run_id.get()
        extra_headers: dict[str, str] = {"X-Auth-Token": self._xauth_token}
        if run_id:
            extra_headers["X-Lex-Run-Id"] = run_id
        stream = await litellm.acompletion(
            model=self.model,
            messages=messages_dicts,
            stream=True,
            api_base=self.base_url,
            api_key="not-needed",
            timeout=30,
            custom_llm_provider="openai",
            extra_headers=extra_headers,
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
