"""Response generation tools with source attribution."""

from typing import AsyncGenerator, Dict, Any, List, Callable
from ..api.event_emitter import EventEmitter
from ..api.event_models import Source
from ..api.connectors.lex_db_connector import LexArticle
from ..api.connectors.openai_provider import LLMProvider
from .extract_used_sources_via_llm import extract_used_sources_via_llm


def create_response_generation_step(
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

    async def generate_response_with_sources(
        context: Dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str, None]:
        """
        Generates a response using retrieved documents, enforces strict factual grounding,
        and emits only the sources actually used in the response.
        """
        retrieved_docs: List[LexArticle] = context.get("retrieved_docs", [])
        user_input: str = context.get("user_input", "").strip()
        conversation_history: List[Dict[str, str]] = context.get(
            "conversation_history", []
        )

        if not retrieved_docs:
            # If no documents were retrieved, defer immediately
            yield emitter.text_chunk(deferral_message)
            context["final_response"] = deferral_message
            return

        # Format retrieved documents for the prompt
        docs_text = "\n\n".join(
            [f"Titel: {doc.title}\nIndhold: {doc.text}" for doc in retrieved_docs]
        )

        sources = f"""
## Artikler (hentet fra Lex)
{docs_text}
"""

        # Always include sources in the user message with clear delineation
        user_message_with_sources = (
            f"{sources}\n---\n\n**Brugerens spørgsmål:**\n{user_input}"
        )
        context["user_message_with_sources"] = user_message_with_sources

        # Prepare messages
        if not conversation_history:
            # First query: include system prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message_with_sources},
            ]
            context["system_prompt"] = system_prompt
        else:
            # Follow-up questions: append to conversation history
            messages = conversation_history + [
                {"role": "user", "content": user_message_with_sources}
            ]

        # Stream response from LLM
        full_response = ""
        async for chunk in llm_provider.generate_stream(messages):  # type: ignore
            full_response += chunk
            yield emitter.text_chunk(chunk)
        context["final_response"] = full_response

        # Identify which sources were actually used
        used_sources = await extract_used_sources_via_llm(
            response=full_response,
            retrieved_docs=retrieved_docs,
            llm_provider=llm_provider,
        )
        context["sources"] = used_sources
        # Emit only the used sources (not all retrieved ones)
        yield emitter.sources(
            [Source(id=src.id, title=src.title, url=src.url) for src in used_sources]
        )

    return generate_response_with_sources
