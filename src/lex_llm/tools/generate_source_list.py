"""Source list generation step."""

from collections.abc import AsyncGenerator, Callable
from typing import Any

from ..api.event_emitter import EventEmitter
from ..api.connectors.openai_provider import LLMProvider
from ..api.connectors.lex_db_connector import LexArticle
from ..api.event_models import ConversationMessage, Source
from ..prompts_search_synthesis import get_source_attribution_prompt
from .llm_json import parse_json_response


def generate_source_list(
    llm_provider: LLMProvider,
) -> Callable[[dict[str, Any], EventEmitter], AsyncGenerator[str | None, None]]:
    """Creates a step that identifies which sources were used in the answer.

    Uses LLM analysis to determine which retrieved documents were actually
    referenced in the answer body, then emits the source list and composes
    the final response for conversation history.

    Sets context keys:
        - used_sources: list[Dict] — the used source articles
        - system_prompt: str — system prompt with sources for conversation history
        - final_response: str — the complete structured answer for history
    """

    async def _generate_source_list(
        context: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str | None, None]:
        # Skip if workflow is done
        if context.get("_workflow_done"):
            return

        answer_body: str = context.get("answer_body", "")
        retrieved_docs: list[LexArticle] = context.get("retrieved_docs", [])
        interpretation: str = context.get("query_interpretation", "")
        lead_paragraph: str = context.get("lead_paragraph", "")
        definitions = context.get("definitions", [])
        system_prompt_base: str = context.get("system_prompt_base", "")

        if not answer_body or not retrieved_docs:
            # No sources to attribute
            context["used_sources"] = []
            context["final_response"] = _compose_final_response(
                interpretation=interpretation,
                lead_paragraph=lead_paragraph,
                answer_body=answer_body,
                definitions=definitions,
            )
            return

        # Format source descriptions for the LLM
        source_descriptions = "\n".join(
            [
                f"ID: {doc.id} | Titel: {doc.title} | Indhold: {doc.text[:300]}..."
                if len(doc.text) > 300
                else f"ID: {doc.id} | Titel: {doc.title} | Indhold: {doc.text}"
                for doc in retrieved_docs
            ]
        )

        messages = get_source_attribution_prompt(
            response=answer_body,
            retrieved_docs_summary=source_descriptions,
        )

        llm_messages = [
            ConversationMessage(role=m["role"], content=m["content"])  # type: ignore
            for m in messages
        ]

        raw_response = await llm_provider.generate(llm_messages)

        try:
            result = parse_json_response(raw_response)
            used_ids = [str(sid) for sid in result.get("source_ids", [])]
        except ValueError:
            used_ids = []

        # Filter retrieved docs by matched IDs
        used_docs = [doc for doc in retrieved_docs if str(doc.id) in used_ids]

        # Store used sources as dicts for conversation history
        used_sources_data = [
            {
                "id": str(doc.id),
                "title": doc.title,
                "text": doc.text,
                "url": doc.url,
            }
            for doc in used_docs
        ]
        context["used_sources"] = used_sources_data

        # Build system prompt with sources for conversation history
        system_prompt_with_sources = system_prompt_base
        if used_sources_data:
            artikler_text = "\n\n".join(
                [
                    f"Titel: {src['title']}\nIndhold: {src['text']}\nURL: {src['url']}\nID: {src['id']}"
                    for src in used_sources_data
                ]
            )
            system_prompt_with_sources += f"\n\n## Artikler\n{artikler_text}"

        context["system_prompt"] = system_prompt_with_sources

        # Emit sources
        yield emitter.sources(
            [
                Source(id=doc.id, title=doc.title, url=doc.url if doc.url else "")
                for doc in used_docs
            ]
        )

        # Compose final response for conversation history
        context["final_response"] = _compose_final_response(
            interpretation=interpretation,
            lead_paragraph=lead_paragraph,
            answer_body=answer_body,
            definitions=definitions,
        )

    return _generate_source_list


def _compose_final_response(
    interpretation: str,
    lead_paragraph: str,
    answer_body: str,
    definitions: list,
) -> str:
    """Compose the final structured response for conversation history.

    This is stored as the assistant message in the conversation history
    so that follow-up queries can reference all sections.
    """
    sections = []

    if interpretation:
        sections.append(f"## Fortolkning\n{interpretation}")

    if lead_paragraph:
        sections.append(f"## Resumé\n{lead_paragraph}")

    if answer_body:
        sections.append(f"## Svar\n{answer_body}")

    if definitions:
        def_lines = [f"**{d.term}**: {d.definition}" for d in definitions]
        sections.append("## Definitioner\n" + "\n".join(def_lines))

    return "\n\n".join(sections)
