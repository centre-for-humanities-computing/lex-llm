"""Response generation with source attribution — v2 (clean history).

Variant of ``generate_response_with_sources`` that injects retrieved
sources into the user message instead of rewriting the system prompt.

Use this in workflows that opt into ``use_clean_history=True``.
"""

from collections.abc import AsyncGenerator, Callable
from typing import Any
from ..api.event_emitter import EventEmitter
from ..api.event_models import Source, ConversationMessage
from ..api.connectors.lex_db_connector import LexArticle
from ..api.connectors.openai_provider import LLMProvider
from .extract_used_sources_via_llm import extract_used_sources_via_llm
from .source_formatting import build_user_message_with_sources


def generate_response_with_sources_v2(
    llm_provider: LLMProvider,
    system_prompt: str,
    deferral_message: str,
) -> tuple[Callable[[dict[str, Any], EventEmitter], AsyncGenerator[str, None]], str]:
    """Same contract as ``generate_response_with_sources`` but with clean-history semantics.

    Sources are appended to the user message under a ``# Kilder`` heading.
    The system prompt is sent verbatim. On the first turn,
    ``context["system_prompt"]`` is set to the base prompt.
    """

    async def generate_response_with_sources_v2(
        context: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str, None]:
        retrieved_docs: list[LexArticle] = context.get("retrieved_docs", [])
        user_input: str = context.get("user_input", "").strip()
        insufficient_context: bool = context.get("insufficient_context", False)
        insufficient_context_reason: str = context.get(
            "insufficient_context_reason", "N/A"
        )
        conversation_history: list[ConversationMessage] = context.get(
            "conversation_history", []
        )

        if not retrieved_docs:
            yield emitter.text_chunk(deferral_message)
            context["final_response"] = deferral_message
            return

        if insufficient_context:
            detailed_deferral = (
                f"{deferral_message}\n\nÅrsag: {insufficient_context_reason}"
            )
            yield emitter.text_chunk(detailed_deferral)
            context["final_response"] = detailed_deferral
            return

        # Build user message with retrieved sources appended
        user_message_with_sources = build_user_message_with_sources(
            user_input=user_input,
            retrieved_docs=retrieved_docs,
        )

        # Build messages: stable system prompt + history + user with sources
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        for msg in conversation_history:
            if dict(msg)["role"] in ("user", "assistant"):
                messages.append(
                    {"role": dict(msg)["role"], "content": dict(msg)["content"]}
                )
        messages.append({"role": "user", "content": user_message_with_sources})

        messages_for_llm: list[ConversationMessage] = [
            ConversationMessage(role=m["role"], content=m["content"])  # type: ignore
            for m in messages
        ]

        telemetry = context.get("_current_step_telemetry", {})

        full_response = ""
        async with llm_provider.observe(telemetry=telemetry):
            async for chunk in llm_provider.generate_stream(messages_for_llm):  # type: ignore
                full_response += chunk
                yield emitter.text_chunk(chunk)
        context["final_response"] = full_response

        # Set the base system prompt for conversation history (first turn only)
        if not conversation_history:
            context["system_prompt"] = system_prompt

        # Extract which sources were actually used
        async with llm_provider.observe(telemetry=telemetry):
            newly_used_sources = await extract_used_sources_via_llm(
                response=full_response,
                retrieved_docs=retrieved_docs,
                llm_provider=llm_provider,
            )

        used_sources_data = [
            {
                "id": str(src.id),
                "title": src.title,
                "text": src.text,
                "url": src.url if src.url else "",
            }
            for src in newly_used_sources
        ]
        context["used_sources"] = used_sources_data

        yield emitter.sources(
            [
                Source(
                    id=src.id,
                    title=src.title,
                    url=src.url if src.url else "",
                    highlight=src.highlight,
                )
                for src in newly_used_sources
            ]
        )

    return generate_response_with_sources_v2, "Skriver svar ud fra fundne kilder"
