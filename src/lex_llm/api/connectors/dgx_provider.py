# llm/dgx_provider.py
import os
import time
from typing import AsyncGenerator, List
import litellm
from ..event_models import ConversationMessage
from .llm_provider import LLMProvider


class DGXOverloadedError(RuntimeError):
    """Raised when a DGX request is taking too long and should be failed over."""


class DGXProvider(LLMProvider):
    """Talks to the DGX Spark via the nginx-fronted vLLM OpenAI endpoint."""

    def __init__(
        self,
        model: str = "gemma-4-26B-A4B-it",
        base_url: str | None = None,
        *,
        ttft_timeout: float = 8.0,      # seconds to first token
        min_tokens_per_sec: float = 8.0, # abort if stream is slower than this
        check_after_tokens: int = 32,    # don't measure tps until N tokens seen
    ):
        self.model = model
        self.base_url = base_url or os.environ["DGX_BASE_URL"]
        self.ttft_timeout = ttft_timeout
        self.min_tokens_per_sec = min_tokens_per_sec
        self.check_after_tokens = check_after_tokens

    async def generate_stream(
        self, messages: List[ConversationMessage]
    ) -> AsyncGenerator[str, None]:
        start = time.monotonic()
        first_token_at: float | None = None
        token_count = 0

        stream = await litellm.acompletion(
            model="openai/" + self.model,
            messages=messages,
            stream=True,
            api_base=self.base_url,
            api_key="not-needed",  # vLLM doesn't require one by default
            timeout=self.ttft_timeout + 2,
        )

        async for chunk in stream:  # type: ignore
            content = chunk.choices[0].delta.content
            if not content:
                continue

            now = time.monotonic()
            if first_token_at is None:
                first_token_at = now
                if (first_token_at - start) > self.ttft_timeout:
                    raise DGXOverloadedError(
                        f"TTFT {first_token_at - start:.1f}s exceeded {self.ttft_timeout}s"
                    )

            token_count += 1
            # Approximate tokens by chunks; for OpenAI streaming this is usually 1 token per chunk
            if token_count == self.check_after_tokens:
                elapsed = now - first_token_at
                tps = token_count / max(elapsed, 1e-6)
                if tps < self.min_tokens_per_sec:
                    raise DGXOverloadedError(
                        f"Generation speed {tps:.1f} tok/s below {self.min_tokens_per_sec}"
                    )

            yield content

    async def generate(self, messages: List[ConversationMessage]) -> str:
        out = ""
        async for chunk in self.generate_stream(messages):
            out += chunk
        return out