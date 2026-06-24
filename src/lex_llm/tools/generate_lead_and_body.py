"""Merged lead paragraph + answer body generation step (fast workflow).

Generates both the lead and the elaborating body in a single LLM call.
The lead is bold Markdown (``**...**``), followed by a blank line and
the body. Everything streams as ``text_chunk`` events — no separate
``lead_paragraph`` or ``answer_body`` event types.

Retrieved sources are injected into the user message (not the system
prompt) so the system prompt stays stable for KV-cache reuse.
"""

from collections.abc import AsyncGenerator, Callable
from typing import Any

from ..api.event_emitter import EventEmitter
from ..api.connectors.openai_provider import LLMProvider
from ..api.connectors.lex_db_connector import (
    LexArticle,
    LexChunk,
)
from ..api.event_models import ConversationMessage
from ..prompts_search_synthesis import (
    get_insufficient_context_deferral_prompt,
)
from .source_formatting import build_user_message_with_sources


def generate_lead_and_body(
    llm_provider: LLMProvider,
    system_prompt: str,
) -> tuple[
    Callable[[dict[str, Any], EventEmitter], AsyncGenerator[str | None, None]], str
]:
    """Creates a step that generates a bold lead paragraph followed by an
    elaborating answer body in a single LLM streaming call.

    Everything is emitted via ``text_chunk`` events. The lead appears first
    (in Markdown bold), then a blank line, then the body.

    If context indicates insufficient search results, a deferral message
    is generated instead.

    Sets context keys:
        - final_response: str — the complete streamed text
        - answer_body: str — alias for final_response (compatibility
          with generate_source_list._compose_final_response)
        - lead_paragraph: str — set to "" for compatibility
        - system_prompt: str — the base system prompt (first turn only)
    """

    async def _generate_lead_and_body(
        context: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str | None, None]:
        if context.get("_workflow_done"):
            return

        user_input: str = context.get("user_input", "")
        interpretation: str = context.get("query_interpretation", user_input)
        retrieved_chunks: list[LexChunk] = context.get("retrieved_chunks", [])
        retrieved_docs: list[LexArticle] = context.get("retrieved_docs", [])
        conversation_history: list[ConversationMessage] = context.get(
            "conversation_history", []
        )

        # --- Handle insufficient context ---
        if context.get("insufficient_context"):
            insufficient_reason: str = context.get(
                "insufficient_context_reason", "Utilstrækkeligt materiale"
            )
            partial_findings = None
            if retrieved_docs:
                partial_findings = ", ".join(doc.title for doc in retrieved_docs)

            deferral_messages = get_insufficient_context_deferral_prompt(
                user_input=user_input,
                interpretation=interpretation,
                insufficient_context_reason=insufficient_reason,
                partial_findings=partial_findings,
            )
            llm_deferral_messages = [
                ConversationMessage(role=m["role"], content=m["content"])  # type: ignore
                for m in deferral_messages
            ]

            telemetry = context.get("_current_step_telemetry", {})

            deferral_text = ""
            async with llm_provider.observe(telemetry=telemetry):
                async for chunk in llm_provider.generate_stream(llm_deferral_messages):  # type: ignore
                    yield emitter.text_chunk(chunk)
                    deferral_text += chunk

            context["final_response"] = deferral_text
            context["answer_body"] = deferral_text
            context["lead_paragraph"] = ""
            context["_workflow_done"] = True
            return

        # --- Build user message with sources ---
        user_message_with_sources = build_user_message_with_sources(
            user_input=user_input,
            retrieved_chunks=retrieved_chunks,
        )

        # --- Build messages: stable system prompt + history + user with sources ---
        messages: list[ConversationMessage] = [
            ConversationMessage(role="system", content=system_prompt),
        ]
        for msg in conversation_history:
            msg_dict = dict(msg) if not isinstance(msg, dict) else msg
            if msg_dict.get("role") in ("user", "assistant"):
                messages.append(
                    ConversationMessage(
                        role=msg_dict["role"],  # type: ignore
                        content=msg_dict["content"],
                    )
                )
        messages.append(
            ConversationMessage(role="user", content=user_message_with_sources)
        )

        # --- Stream response ---
        full_response = ""

        telemetry = context.get("_current_step_telemetry", {})

        async with llm_provider.observe(telemetry=telemetry):
            async for chunk in llm_provider.generate_stream(messages):  # type: ignore
                full_response += chunk
                yield emitter.text_chunk(chunk)

        context["final_response"] = full_response
        context["answer_body"] = full_response
        context["lead_paragraph"] = ""

        # Set the base system prompt for conversation history (first turn only)
        if not conversation_history:
            context["system_prompt"] = system_prompt

    return _generate_lead_and_body, "Genererer svar ud fra de fundne kilder"
