"""Deferral generation step for out-of-scope queries."""

from collections.abc import AsyncGenerator, Callable
from typing import Any

from ..api.event_emitter import EventEmitter
from ..api.connectors.openai_provider import LLMProvider
from ..api.event_models import ConversationMessage
from ..prompts_search_synthesis import get_deferral_prompt


def generate_deferral(
    llm_provider: LLMProvider,
) -> tuple[
    Callable[[dict[str, Any], EventEmitter], AsyncGenerator[str | None, None]], str
]:
    """Creates a step that generates a deferral message for out-of-scope queries.

    If the query is in scope (context["is_in_scope"] is True), this step
    is a no-op. If out of scope, it generates a deferral message and
    signals early termination.

    Sets context keys (when out of scope):
        - final_response: str — the deferral message
        - _workflow_done: True
    """

    async def _generate_deferral(
        context: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str | None, None]:
        is_in_scope: bool = context.get("is_in_scope", True)

        if is_in_scope:
            # Query is in scope — nothing to do
            return

        user_input: str = context.get("user_input", "")
        routing_reason: str = context.get("routing_reason", "")

        messages = [
            ConversationMessage(role=m["role"], content=m["content"])  # type: ignore
            for m in get_deferral_prompt(
                user_input=user_input,
                routing_reason=routing_reason,
            )
        ]

        telemetry = context.get("_current_step_telemetry", {})

        deferral_message = ""
        async with llm_provider.observe(telemetry=telemetry):
            async for chunk in llm_provider.generate_stream(messages):  # type: ignore
                deferral_message += chunk
                yield emitter.text_chunk(chunk)

        # Emit as lead paragraph
        yield emitter.lead_paragraph_chunk(deferral_message)

        context["final_response"] = deferral_message
        context["_workflow_done"] = True

    return _generate_deferral, "Vurderer brugerens forespørgsel"
