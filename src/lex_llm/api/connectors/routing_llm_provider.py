# llm/routing_provider.py
import logging
from typing import AsyncGenerator, List
from ..event_models import ConversationMessage
from .llm_provider import LLMProvider
from .vllm_load_probe import VLLMLoadProbe

logger = logging.getLogger(__name__)


class RoutingLLMProvider(LLMProvider):
    """Routes requests to a local vLLM backend, falling back to a cloud
    provider when the local backend is overloaded or unreachable.

    Decision flow
    -------------
    1. Probe the vLLM ``/metrics`` endpoint.  If overloaded (queue
       depth or token generation speed exceed thresholds), stream
       directly from the fallback.
    2. Otherwise try the primary.  If it raises **before** yielding
       the first token, catch the error, log it, and stream from the
       fallback instead.
    3. If the primary fails **after** at least one token has been
       yielded, the exception propagates (no mid-stream failover).
    """

    def __init__(
        self,
        primary: LLMProvider,
        fallback: LLMProvider,
        probe: VLLMLoadProbe,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.probe = probe

    async def generate_stream(
        self, messages: List[ConversationMessage]
    ) -> AsyncGenerator[str, None]:
        overloaded, reason = await self.probe.is_overloaded()
        if overloaded:
            logger.info("Routing to fallback: %s", reason)
            async for chunk in self.fallback.generate_stream(messages):
                yield chunk
            return
        stream = self.primary.generate_stream(messages)
        try:
            first = await stream.__anext__()
        except StopAsyncIteration:
            # Primary produced no tokens — benign empty response.
            return
        except Exception:
            logger.exception("Primary failed before first token; failing over")
            async for chunk in self.fallback.generate_stream(messages):
                yield chunk
            return

        # Primary succeeded — yield the first chunk, then the rest.
        yield first
        async for chunk in stream:
            yield chunk

    async def generate(self, messages: List[ConversationMessage]) -> str:
        out = ""
        async for chunk in self.generate_stream(messages):
            out += chunk
        return out
