"""Response generation with inline citations — v3 (clean history).

Variant of ``generate_response_with_sources_v2`` that replaces the explicit
source-attribution LLM call with inline ``[^ID]`` citation markers. Sources
are extracted from the generated text via regex post-processing — no second
LLM call required.

Citation markers are stripped during streaming so the client never sees
them. The raw (unstripped) text is held in memory for source extraction.

Use this in workflows that opt into ``use_clean_history=True``.
"""

from collections.abc import AsyncGenerator, Callable
from typing import Any
from ..api.event_emitter import EventEmitter
from ..api.event_models import Source, ConversationMessage
from ..api.connectors.lex_db_connector import LexArticle
from ..api.connectors.openai_provider import LLMProvider
from .source_formatting import build_user_message_with_sources
from .citation_extraction import extract_cited_sources, CitationStripper


def generate_response_with_sources_v3(
    llm_provider: LLMProvider,
    system_prompt: str,
    deferral_message: str,
    *,
    current_date: str | None = None,
) -> tuple[Callable[[dict[str, Any], EventEmitter], AsyncGenerator[str, None]], str]:
    """Same contract as ``generate_response_with_sources_v2`` but uses inline
    ``[^ID]`` citation markers instead of a separate attribution LLM call.

    After streaming the response, citation markers are extracted via regex
    and mapped to ``LexArticle`` objects for the ``sources`` event and
    ``used_sources`` context key.
    """

    async def _generate_response_with_sources_v3(
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

        # Build user message with retrieved sources and date appended
        user_message_with_sources = build_user_message_with_sources(
            user_input=user_input,
            retrieved_docs=retrieved_docs,
            current_date=current_date,
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

        # --- Stream response with citation stripping ---
        raw_response = ""
        clean_response = ""
        stripper = CitationStripper()

        telemetry = context.get("_current_step_telemetry", {})

        async with llm_provider.observe(telemetry=telemetry):
            async for chunk in llm_provider.generate_stream(messages_for_llm):  # type: ignore
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

        # final_response = clean (client-facing / conversation history)
        context["final_response"] = clean_response

        # Set the base system prompt for conversation history (first turn only)
        if not conversation_history:
            context["system_prompt"] = system_prompt

        # Extract cited sources from raw (unstripped) response
        yield emitter.tool_call(
            name="citation_extraction",
            input_data={
                "num_sources": len(retrieved_docs),
                "source_ids": [doc.id for doc in retrieved_docs],
                "method": "regex [^ID]",
            },
            description="Udtrækker [^ID]-citationer fra svaret...",
        )

        newly_used_sources = extract_cited_sources(
            response=raw_response,
            retrieved_docs=retrieved_docs,
        )

        yield emitter.tool_result(
            name="citation_extraction",
            result_data={
                "used_ids": [doc.id for doc in newly_used_sources],
                "num_used": len(newly_used_sources),
                "total_retrieved": len(retrieved_docs),
                "fallback_used": "[^" not in raw_response,  # True if no markers were found
            },
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

    return _generate_response_with_sources_v3, "Skriver svar ud fra fundne kilder"
