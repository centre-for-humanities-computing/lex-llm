"""Source list generation step — v2 (clean history).

Variant of ``generate_source_list`` that does not touch
``context["system_prompt"]``. The system prompt is owned by the upstream
generation step (e.g. ``generate_lead_and_body_v2``).

Use this in workflows that opt into ``use_clean_history=True``.
"""

from collections.abc import AsyncGenerator, Callable
from typing import Any

from ..api.event_emitter import EventEmitter
from ..api.connectors.openai_provider import LLMProvider
from ..api.connectors.lex_db_connector import LexArticle
from ..api.event_models import ConversationMessage, Source
from ..prompts_search_synthesis import get_source_attribution_prompt
from .llm_json import parse_json_response


def generate_source_list_v2(
    llm_provider: LLMProvider,
) -> tuple[
    Callable[[dict[str, Any], EventEmitter], AsyncGenerator[str | None, None]], str
]:
    """Same contract as ``generate_source_list`` but does not write system_prompt."""

    async def _generate_source_list_v2(
        context: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str | None, None]:
        if context.get("_workflow_done"):
            return

        answer_body: str = context.get("answer_body", "")
        retrieved_docs: list[LexArticle] = context.get("retrieved_docs", [])
        interpretation: str = context.get("query_interpretation", "")
        lead_paragraph: str = context.get("lead_paragraph", "")
        definitions = context.get("definitions", [])

        if not answer_body or not retrieved_docs:
            context["used_sources"] = []
            context["final_response"] = _compose_final_response(
                interpretation=interpretation,
                lead_paragraph=lead_paragraph,
                answer_body=answer_body,
                definitions=definitions,
            )
            return

        source_descriptions = "\n".join(
            [
                f"ID: {doc.id} | Titel: {doc.title} | Indhold: {doc.text}"
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

        telemetry = context.get("_current_step_telemetry", {})

        async with llm_provider.observe(telemetry=telemetry):
            raw_response = await llm_provider.generate(llm_messages)

        try:
            result = parse_json_response(raw_response)
            used_ids = [str(sid) for sid in result.get("source_ids", [])]
        except ValueError:
            used_ids = []

        used_docs = [doc for doc in retrieved_docs if str(doc.id) in used_ids]

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

        yield emitter.sources(
            [
                Source(id=doc.id, title=doc.title, url=doc.url, highlight=doc.highlight)
                for doc in used_docs
            ]
        )

        context["final_response"] = _compose_final_response(
            interpretation=interpretation,
            lead_paragraph=lead_paragraph,
            answer_body=answer_body,
            definitions=definitions,
        )

    return _generate_source_list_v2, "Skriver kildeliste"


def _compose_final_response(
    interpretation: str,
    lead_paragraph: str,
    answer_body: str,
    definitions: list,
) -> str:
    """Compose the final structured response for conversation history.

    (Duplicated from ``generate_source_list._compose_final_response`` to
    keep this module self-contained; the original is unchanged.)
    """
    sections = []

    if interpretation:
        sections.append(f"## Fortolkning\n{interpretation}")

    if lead_paragraph:
        sections.append(f"## Manchet\n{lead_paragraph}")

    if answer_body:
        sections.append(f"## Svar\n{answer_body}")

    if definitions:
        def_lines = [f"**{d.term}**: {d.definition}" for d in definitions]
        sections.append("## Definitioner\n" + "\n".join(def_lines))

    return "\n\n".join(sections)
