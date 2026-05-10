"""Combined query interpretation and scope routing step."""

from collections.abc import AsyncGenerator, Callable
from typing import Any

from ..api.event_emitter import EventEmitter
from ..api.connectors.openai_provider import LLMProvider
from ..api.event_models import ConversationMessage
from ..prompts_search_synthesis import get_interpret_and_route_prompt
from .llm_json import parse_json_response


def interpret_and_route(
    llm_provider: LLMProvider,
) -> Callable[[dict[str, Any], EventEmitter], AsyncGenerator[str | None, None]]:
    """Creates a step that interprets the user query and routes by scope.

    Uses a single LLM call to both interpret the query and determine
    whether it falls within the encyclopedia's domain.

    Sets context keys:
        - query_interpretation: str — the interpreted query
        - is_in_scope: bool — whether the query is within scope
        - routing_reason: str — reason for the routing decision
        - _workflow_done: set to True if out of scope (early termination)
    """

    async def _interpret_and_route(
        context: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str | None, None]:
        user_input: str = context.get("user_input", "")
        conversation_history = context.get("conversation_history", [])

        # Build conversation history summary for context
        history_summary = None
        if conversation_history:
            parts = []
            for msg in conversation_history:
                msg_dict = dict(msg) if not isinstance(msg, dict) else msg
                if msg_dict.get("role") in ("user", "assistant"):
                    parts.append(f"{msg_dict['role']}: {msg_dict['content']}")
            if parts:
                history_summary = "\n".join(parts)

        messages = get_interpret_and_route_prompt(
            user_input=user_input,
            conversation_history=history_summary,
        )

        # Convert dicts to ConversationMessage for the LLM provider
        llm_messages = [
            ConversationMessage(role=m["role"], content=m["content"])  # type: ignore
            for m in messages
        ]

        raw_response = await llm_provider.generate(llm_messages)

        try:
            result = parse_json_response(raw_response)
            interpretation = result.get("interpretation", user_input)
            in_scope = result.get("in_scope", True)
            reason = result.get("reason", "")
        except ValueError:
            # Fallback: if JSON parsing fails, assume in scope with raw response
            interpretation = raw_response.strip()
            in_scope = True
            reason = "Kunne ikke fortolke routing-svar — antager inden for scope"

        context["query_interpretation"] = interpretation
        context["is_in_scope"] = in_scope
        context["routing_reason"] = reason

        # Emit the interpretation as a stream event
        yield emitter.interpretation_chunk(interpretation)

        # If out of scope, signal early termination
        if not in_scope:
            context["_workflow_done"] = True

    return _interpret_and_route
