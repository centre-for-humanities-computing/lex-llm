"""Merged lead paragraph + answer body generation — v3 (inline citations, stripped).

Variant of ``generate_lead_and_body_v2`` that strips ``[^ID]`` citation
markers during streaming so the client never sees them. The raw
(unstripped) answer body is preserved in context for downstream source
extraction.

Use this in workflows that opt into ``use_clean_history=True``.
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
from .citation_extraction import CitationStripper


def generate_lead_and_body_v3(
    llm_provider: LLMProvider,
    system_prompt: str,
    *,
    current_date: str | None = None,
) -> tuple[
    Callable[[dict[str, Any], EventEmitter], AsyncGenerator[str | None, None]], str
]:
    """Same contract as ``generate_lead_and_body_v2`` but strips ``[^ID]``
    citation markers during streaming.

    The raw (unstripped) body is stored in ``context["_raw_answer_body"]``
    so the downstream source-list step can extract cited sources from the
    original markers. ``context["answer_body"]`` and
    ``context["final_response"]`` contain the cleaned text.

    On the first turn, ``context["system_prompt"]`` is set to the base
    prompt so the orchestrator can persist it as a stable system message.
    """

    async def _generate_lead_and_body_v3(
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

        # --- Build user message with sources and date ---
        user_message_with_sources = build_user_message_with_sources(
            user_input=user_input,
            retrieved_chunks=retrieved_chunks,
            current_date=current_date,
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

        # --- Stream response with citation stripping ---
        raw_response = ""
        clean_response = ""
        stripper = CitationStripper()

        telemetry = context.get("_current_step_telemetry", {})

        async with llm_provider.observe(telemetry=telemetry):
            async for chunk in llm_provider.generate_stream(messages):  # type: ignore
                raw_response += chunk
                clean = stripper.feed(chunk)
                if clean:
                    yield emitter.text_chunk(clean)
                    clean_response += clean

        # Flush any remaining buffered text
        tail = stripper.flush()
        if tail:
            yield emitter.text_chunk(tail)
            clean_response += tail

        # final_response and answer_body = clean (client-facing / history)
        context["final_response"] = clean_response
        context["answer_body"] = clean_response
        context["lead_paragraph"] = ""

        # Preserve raw body for downstream citation extraction
        context["_raw_answer_body"] = raw_response

        # Set the base system prompt for conversation history (first turn only)
        if not conversation_history:
            context["system_prompt"] = system_prompt

    return _generate_lead_and_body_v3, "Genererer svar ud fra de fundne kilder"
