"""Response generation tools with source attribution."""

import re
from typing import AsyncGenerator, Dict, Any, List, Callable
from ..api.event_emitter import EventEmitter
from ..api.event_models import Source, ConversationMessage
from ..api.connectors.lex_db_connector import LexArticle
from ..api.connectors.openai_provider import LLMProvider
from .extract_used_sources_via_llm import extract_used_sources_via_llm


def _extract_used_sources_from_system_prompt(
    conversation_history: List[ConversationMessage],
) -> List[Dict[str, str]]:
    """
    Extract used sources from the system prompt in conversation history.
    Returns a list of dicts with id, title, text, and url.
    """
    if not conversation_history:
        return []

    # Find the system message
    system_message = None
    for msg in conversation_history:
        if dict(msg)["role"] == "system":
            system_message = dict(msg)["content"]
            break

    if not system_message:
        return []

    # Extract the "Artikler" section using regex
    artikler_match = re.search(
        r"## Artikler\n(.+?)(?=\n## |$)", system_message, re.DOTALL
    )

    if not artikler_match:
        return []

    artikler_section = artikler_match.group(1)

    # Extract individual articles
    # Pattern: Titel: <title>\nIndhold: <text>\nURL: <url>\nID: <id>
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


def generate_response_with_sources(
    llm_provider: LLMProvider,
    system_prompt: str,
    deferral_message: str,
) -> Callable[[Dict[str, Any], EventEmitter], AsyncGenerator[str, None]]:
    """
    Creates a response generation step with source attribution.

    Args:
        llm_provider: The LLM provider to use for generation
        system_prompt: The system prompt to guide the LLM
        deferral_message: Message to return when no documents are available

    Returns:
        An async generator function compatible with the Orchestrator
    """

    async def run(
        context: Dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str, None]:
        """
        Generates a response using retrieved documents, enforces strict factual grounding,
        and emits only the sources actually used in the response.
        """
        retrieved_docs: List[LexArticle] = context.get("retrieved_docs", [])
        user_input: str = context.get("user_input", "").strip()
        conversation_history: List[ConversationMessage] = context.get(
            "conversation_history", []
        )

        if not retrieved_docs:
            # If no documents were retrieved, defer immediately
            yield emitter.text_chunk(deferral_message)
            context["final_response"] = deferral_message
            return

        # Extract previously used sources from conversation history (stateless)
        previous_used_sources_data = _extract_used_sources_from_system_prompt(
            conversation_history
        )

        # Build dynamic system prompt with sources
        dynamic_system_prompt = system_prompt

        # Add "Artikler" section if there are previously used sources
        if previous_used_sources_data:
            artikler_text = "\n\n".join(
                [
                    f"Titel: {src['title']}\nIndhold: {src['text']}\nURL: {src['url']}\nID: {src['id']}"
                    for src in previous_used_sources_data
                ]
            )
            dynamic_system_prompt += f"\n\n# Artikler\n{artikler_text}"

        # Always add "Potentielle kilder" section with new retrieved docs
        potentielle_text = "\n\n".join(
            [
                f"Titel: {doc.title}\nIndhold: {doc.text}\nURL: {doc.url}\nID: {doc.id}"
                for doc in retrieved_docs
            ]
        )
        dynamic_system_prompt += f"\n\n# Potentielle kilder\n{potentielle_text}"

        # Prepare messages with dynamic system prompt (as dicts for LLM provider)
        messages: List[Dict[str, str]] = []

        if not conversation_history:
            # First query: include dynamic system prompt with Potentielle kilder
            messages = [
                {"role": "system", "content": dynamic_system_prompt},
                {"role": "user", "content": user_input},
            ]
        else:
            # Follow-up: rebuild with dynamic system prompt (Artikler + Potentielle kilder)
            messages = [
                {"role": "system", "content": dynamic_system_prompt},
            ]
            # Add conversation history (user/assistant pairs only)
            for msg in conversation_history:
                if dict(msg)["role"] in ["user", "assistant"]:
                    messages.append(
                        {"role": dict(msg)["role"], "content": dict(msg)["content"]}
                    )
            # Add new user message
            messages.append({"role": "user", "content": user_input})

        # Stream response from LLM
        full_response = ""
        async for chunk in llm_provider.generate_stream(messages):  # type: ignore
            full_response += chunk
            yield emitter.text_chunk(chunk)
        context["final_response"] = full_response

        # Extract which NEW sources (from Potentielle kilder) were actually used
        newly_used_sources = await extract_used_sources_via_llm(
            response=full_response,
            retrieved_docs=retrieved_docs,
            llm_provider=llm_provider,
        )

        # Merge with previous used sources, avoiding duplicates by ID
        previous_ids = {src["id"] for src in previous_used_sources_data}
        merged_used_sources = previous_used_sources_data.copy()

        for src in newly_used_sources:
            if str(src.id) not in previous_ids:
                merged_used_sources.append(
                    {
                        "id": str(src.id),
                        "title": src.title,
                        "text": src.text,
                        "url": src.url,
                    }
                )
                previous_ids.add(str(src.id))

        # Store merged sources for building system prompt in next turn
        context["used_sources"] = merged_used_sources

        # Build the system prompt with "Artikler" section for conversation history
        system_prompt_with_sources = system_prompt
        if merged_used_sources:
            artikler_text = "\n\n".join(
                [
                    f"Titel: {src['title']}\nIndhold: {src['text']}\nURL: {src['url']}\nID: {src['id']}"
                    for src in merged_used_sources
                ]
            )
            system_prompt_with_sources += f"\n\n## Artikler\n{artikler_text}"

        context["system_prompt"] = system_prompt_with_sources

        # Emit only the newly used sources (not the ones already in Artikler)
        yield emitter.sources(
            [
                Source(id=src.id, title=src.title, url=src.url)
                for src in newly_used_sources
            ]
        )

    return run
