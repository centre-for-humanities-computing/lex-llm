# llm/routing_provider.py
import contextvars
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, List
from ..event_models import ConversationMessage
from .llm_provider import LLMProvider, RouteDecision
from .vllm_load_probe import VLLMLoadProbe

logger = logging.getLogger(__name__)

# Per-task context variable for capturing routing decisions
_route_callback: contextvars.ContextVar[Callable[[RouteDecision], None] | None] = (
    contextvars.ContextVar("_route_callback", default=None)
)

# Per-task reference to the current llm_call entry being populated, so
# generate_stream / generate can update token counts after completion.
_current_call_entry: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("_current_call_entry", default=None)
)


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

    # ── observability hook ───────────────────────────────────────────

    @asynccontextmanager
    async def observe(
        self,
        callback: "Callable[[RouteDecision], None] | None" = None,
        *,
        telemetry: "dict[str, Any] | None" = None,
    ) -> "AsyncGenerator[None, None]":
        """Context manager that captures routing + token counts.

        Two modes:

        *callback* — called with each ``RouteDecision`` as it is made.
        *telemetry* — a ``dict`` to which per-LLM-call entries are appended
        automatically under ``"llm_calls"``.  Each entry includes
        ``input_chars`` and ``output_chars`` (character-level estimates
        of prompt and completion size).

        If both are given, *callback* takes precedence.

        Usage::

            async with llm_provider.observe(telemetry=step_telemetry):
                await llm_provider.generate(messages)
        """
        effective_cb: Callable[[RouteDecision], None] | None = callback
        # Build a preliminary entry if using the telemetry shorthand
        preliminary_entry: dict[str, Any] | None = None
        if effective_cb is None and telemetry is not None:
            preliminary_entry = {"input_chars": 0, "output_chars": 0}
            telemetry.setdefault("llm_calls", []).append(preliminary_entry)

            def _recorder(decision: RouteDecision) -> None:
                preliminary_entry.update(
                    {
                        "backend": decision.backend,
                        "trigger": decision.trigger,
                        "reason": decision.reason,
                        "model": decision.model,
                    }
                )

            effective_cb = _recorder

        route_token = _route_callback.set(effective_cb)
        call_token = _current_call_entry.set(preliminary_entry)
        try:
            yield
        finally:
            _route_callback.reset(route_token)
            _current_call_entry.reset(call_token)

    def _fire_route_callback(
        self,
        backend: str,
        trigger: str,
        reason: str = "",
        model: str = "",
    ) -> None:
        cb = _route_callback.get()
        if cb is not None:
            cb(
                RouteDecision(
                    backend=backend,  # type: ignore[arg-type]
                    trigger=trigger,  # type: ignore[arg-type]
                    reason=reason,
                    model=model,
                )
            )

    # ── inference ────────────────────────────────────────────────────

    async def generate_stream(
        self, messages: List[ConversationMessage]
    ) -> AsyncGenerator[str, None]:
        # Estimate input size for token-counting
        input_chars = sum(len(m.content) for m in messages)
        entry = _current_call_entry.get()
        if entry is not None:
            entry["input_chars"] = input_chars

        overloaded, reason = await self.probe.is_overloaded()
        if overloaded:
            trigger = (
                "probe_scrape_error"
                if reason.startswith("scrape error")
                else "probe_overload"
            )
            self._fire_route_callback(
                backend="fallback",
                trigger=trigger,
                reason=reason,
                model=getattr(self.fallback, "model", ""),
            )
            logger.info("Routing to fallback: %s", reason)
            output_chars = 0
            async for chunk in self.fallback.generate_stream(messages):
                output_chars += len(chunk)
                yield chunk
            if entry is not None:
                entry["output_chars"] = output_chars
            return

        stream = self.primary.generate_stream(messages)
        try:
            first = await stream.__anext__()
        except StopAsyncIteration:
            # Primary produced no tokens — benign empty response.
            if entry is not None:
                entry["output_chars"] = 0
            return
        except Exception as exc:
            logger.exception("Primary failed before first token; failing over")
            self._fire_route_callback(
                backend="fallback",
                trigger="primary_pre_first_token_error",
                reason=str(exc),
                model=getattr(self.fallback, "model", ""),
            )
            output_chars = 0
            async for chunk in self.fallback.generate_stream(messages):
                output_chars += len(chunk)
                yield chunk
            if entry is not None:
                entry["output_chars"] = output_chars
            return

        # Primary succeeded — yield the first chunk, then the rest.
        self._fire_route_callback(
            backend="primary",
            trigger="ok",
            reason="ok",
            model=getattr(self.primary, "model", ""),
        )
        output_chars = len(first)
        yield first
        async for chunk in stream:
            output_chars += len(chunk)
            yield chunk
        if entry is not None:
            entry["output_chars"] = output_chars

    async def generate(self, messages: List[ConversationMessage]) -> str:
        # generate_stream does the counting; we just need to make sure
        # the call entry is visible.
        out = ""
        async for chunk in self.generate_stream(messages):
            out += chunk
        return out
