"""Response generation tools with source attribution."""

from collections.abc import AsyncGenerator, Callable
from typing import Any
from ..api.event_emitter import EventEmitter
from ..api.event_models import Source, ConversationMessage
from ..api.connectors.lex_db_connector import LexArticle
from ..api.connectors.openai_provider import LLMProvider
from .extract_used_sources_via_llm import extract_used_sources_via_llm
from .source_formatting import build_user_message_with_sources


def generate_response_with_sources(
    llm_provider: LLMProvider,
    system_prompt: str,
    deferral_message: str,
) -> tuple[Callable[[dict[str, Any], EventEmitter], AsyncGenerator[str, None]], str]:
    """
    Creates a response generation step with source attribution.

    Args:
        llm_provider: The LLM provider to use for generation
        system_prompt: The system prompt to guide the LLM
        deferral_message: Message to return when no documents are available

    Returns:
        An async generator function compatible with the Orchestrator
    """

    async def generate_response_with_sources(
        context: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str, None]:
        """
        Generates a response using retrieved documents, enforces strict factual grounding,
        and emits only the sources actually used in the response.
        """
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
            # If no documents were retrieved, defer immediately
            yield emitter.text_chunk(deferral_message)
            context["final_response"] = deferral_message
            return

        if insufficient_context:
            # If there is insufficient context, generate a specific deferral message
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

        # Build messages: stable system prompt + history + user message with sources
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

        # Observability: capture routing decision
        telemetry = context.get("_current_step_telemetry", {})

        # Stream response from LLM
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

        # Store used sources for this turn
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

        # Emit the used sources
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

    return generate_response_with_sources, "Skriver svar ud fra fundne kilder"
