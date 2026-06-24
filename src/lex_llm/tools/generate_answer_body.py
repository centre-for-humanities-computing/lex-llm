"""Answer body generation step.

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


def generate_answer_body(
    llm_provider: LLMProvider,
    system_prompt: str,
) -> tuple[
    Callable[[dict[str, Any], EventEmitter], AsyncGenerator[str | None, None]], str
]:
    """Creates a step that generates the answer body from retrieved documents.

    If context indicates insufficient search results, generates a deferral
    message instead.

    Sets context keys:
        - answer_body: str — the generated answer body text
        - final_response: str — set only if insufficient context (deferral)
        - _workflow_done: set to True if insufficient context
        - system_prompt: str — the base system prompt (first turn only)
    """

    async def _generate_answer_body(
        context: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str | None, None]:
        # Skip if workflow is done
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
            deferral_text = await llm_provider.generate(llm_deferral_messages)
            deferral_text = deferral_text.strip()

            yield emitter.lead_paragraph_chunk(deferral_text)
            context["final_response"] = deferral_text
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
        async for chunk in llm_provider.generate_stream(messages):  # type: ignore
            full_response += chunk
            yield emitter.answer_body_chunk(chunk)

        context["answer_body"] = full_response

        # Set the base system prompt for conversation history (first turn only)
        if not conversation_history:
            context["system_prompt"] = system_prompt

    return _generate_answer_body, "Skriver brødtekst ud fra fundne kilder"
