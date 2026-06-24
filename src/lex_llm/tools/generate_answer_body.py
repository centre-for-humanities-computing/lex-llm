"""Answer body generation step."""

import re
from collections.abc import AsyncGenerator, Callable
from typing import Any

from ..api.event_emitter import EventEmitter
from ..api.connectors.openai_provider import LLMProvider
from ..api.connectors.lex_db_connector import (
    LexArticle,
    LexChunk,
    group_chunks_to_articles,
)
from ..api.event_models import ConversationMessage
from ..prompts_search_synthesis import (
    get_insufficient_context_deferral_prompt,
)


def _extract_used_sources_from_system_prompt(
    conversation_history: list[ConversationMessage],
) -> list[dict[str, str]]:
    """Extract used sources from the system prompt in conversation history.

    Returns a list of dicts with id, title, text, and url.
    """
    if not conversation_history:
        return []

    system_message = None
    for msg in conversation_history:
        msg_dict = dict(msg) if not isinstance(msg, dict) else msg
        if msg_dict.get("role") == "system":
            system_message = msg_dict["content"]
            break

    if not system_message:
        return []

    artikler_match = re.search(
        r"## Artikler\n(.+?)(?=\n## |$)", system_message, re.DOTALL
    )
    if not artikler_match:
        return []

    artikler_section = artikler_match.group(1)
    article_pattern = (
        r"Titel: (.+?)\nIndhold: (.+?)\nURL: (.+?)\nID: (.+?)(?=\n\nTitel: |$)"
    )
    matches = re.finditer(article_pattern, artikler_section, re.DOTALL)

    used_sources = []
    for match in matches:
        used_sources.append(
            {
                "title": match.group(1).strip(),
                "text": match.group(2).strip(),
                "url": match.group(3).strip(),
                "id": match.group(4).strip(),
            }
        )
    return used_sources


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

        # --- Build dynamic system prompt with sources ---
        dynamic_system_prompt = system_prompt

        # Extract previously used sources from conversation history
        previous_used_sources = _extract_used_sources_from_system_prompt(
            conversation_history
        )

        # Add "Artikler" section for previously used sources
        if previous_used_sources:
            artikler_text = "\n\n".join(
                [
                    f"Titel: {src['title']}\nIndhold: {src['text']}\nURL: {src['url']}\nID: {src['id']}"
                    for src in previous_used_sources
                ]
            )
            dynamic_system_prompt += f"\n\n# Artikler\n{artikler_text}"

        # Add "Potentielle kilder" section with new retrieved chunks
        # Sort chunks by (article_id, chunk_seq) to maximize KV cache hits
        sorted_chunks = sorted(
            retrieved_chunks, key=lambda c: (c.article_id, c.chunk_seq)
        )
        # Group sorted chunks into articles for the prompt
        sorted_articles = group_chunks_to_articles(sorted_chunks)
        potentielle_text = "\n\n".join(
            [
                f"Titel: {doc.title}\nIndhold: {doc.text}\nURL: {doc.url}\nID: {doc.id}"
                for doc in sorted_articles
            ]
        )
        dynamic_system_prompt += f"\n\n# Potentielle kilder\n{potentielle_text}"

        # --- Build messages ---
        messages: list[ConversationMessage] = []

        if not conversation_history:
            messages = [
                ConversationMessage(role="system", content=dynamic_system_prompt),
                ConversationMessage(role="user", content=user_input),
            ]
        else:
            messages = [
                ConversationMessage(role="system", content=dynamic_system_prompt),
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
            messages.append(ConversationMessage(role="user", content=user_input))

        # --- Stream response ---
        full_response = ""
        async for chunk in llm_provider.generate_stream(messages):  # type: ignore
            full_response += chunk
            yield emitter.answer_body_chunk(chunk)

        context["answer_body"] = full_response

    return _generate_answer_body, "Skriver brødtekst ud fra fundne kilder..."
