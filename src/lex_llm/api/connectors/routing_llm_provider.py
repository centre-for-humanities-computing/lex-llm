# llm/routing_provider.py
import asyncio
import time
import logging
from typing import AsyncGenerator, List
from ..event_models import ConversationMessage
from .llm_provider import LLMProvider
from .dgx_provider import DGXProvider, DGXOverloadedError
from .scaleway_provider import ScalewayProvider
from ..vllm_health_monitor import VLLMHealthMonitor

logger = logging.getLogger(__name__)

class RoutingLLMProvider(LLMProvider):

    def __init__(
        self,
        primary: LLMProvider,
        fallback: LLMProvider,
        monitor: VLLMHealthMonitor,
        *,
        trip_after: int = 2,        # N overload signals before tripping
        recover_after: int = 5,     # M healthy signals before recovering
    ):
        self.primary = primary
        self.fallback = fallback
        self.monitor = monitor
        self.trip_after = trip_after
        self.recover_after = recover_after

        self._overload_streak = 0
        self._healthy_streak = 0
        self._using_fallback = False
        self._state_lock = asyncio.Lock()

    async def _pick_backend(self) -> tuple[LLMProvider, str]:
        overloaded, reason = await self.monitor.is_overloaded()

        async with self._state_lock:
            if overloaded:
                self._overload_streak += 1
                self._healthy_streak = 0
                if (
                    not self._using_fallback
                    and self._overload_streak >= self.trip_after
                ):
                    self._using_fallback = True
                    logger.warning(f"Tripping to fallback: {reason}")
            else:
                self._healthy_streak += 1
                self._overload_streak = 0
                if (
                    self._using_fallback
                    and self._healthy_streak >= self.recover_after
                ):
                    self._using_fallback = False
                    logger.info("Primary recovered, routing back to it")

            if self._using_fallback:
                return self.fallback, f"fallback ({reason})"
            return self.primary, "primary"

    async def generate_stream(
        self, messages: List[ConversationMessage]
    ) -> AsyncGenerator[str, None]:
        backend, why = await self._pick_backend()
        logger.debug(f"Routing to {backend.__class__.__name__}: {why}")

        # If we picked the primary, be ready to fail over mid-stream.
        if backend is self.primary:
            tokens_yielded = 0
            try:
                async for chunk in backend.generate_stream(messages):
                    tokens_yielded += 1
                    yield chunk
                return
            except DGXOverloadedError as e:
                if tokens_yielded > 0:
                    # We've already sent partial output to the user — restarting on
                    # cloud would produce a discontinuous response. Best to surface
                    # a clear signal and let the caller decide.
                    logger.error(
                        f"Primary degraded mid-stream after {tokens_yielded} tokens: {e}. "
                        f"Cannot transparently fail over."
                    )
                    raise
                logger.warning(f"Primary failed before first token: {e}. Failing over.")
                # Force-trip so subsequent requests skip the probe
                async with self._state_lock:
                    self._using_fallback = True
                    self._overload_streak = self.trip_after
            except Exception as e:
                if tokens_yielded > 0:
                    raise
                logger.exception(f"Primary hard failure: {e}. Failing over.")

        # Fallback path
        async for chunk in self.fallback.generate_stream(messages):
            yield chunk

    async def generate(self, messages: List[ConversationMessage]) -> str:
        out = ""
        async for chunk in self.generate_stream(messages):
            out += chunk
        return out