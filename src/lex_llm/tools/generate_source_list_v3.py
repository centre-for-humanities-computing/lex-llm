"""Source list generation step — v3 (inline citations, clean history).

Variant of ``generate_source_list_v2`` that replaces the explicit
source-attribution LLM call with regex extraction of ``[^ID]``
citation markers from the answer body. No second LLM call is needed.

Use this in editorial workflows that opt into ``use_clean_history=True``.
"""

from collections.abc import AsyncGenerator, Callable
from typing import Any

from ..api.event_emitter import EventEmitter
from ..api.connectors.lex_db_connector import LexArticle
from ..api.event_models import Source
from .citation_extraction import extract_cited_sources


def generate_source_list_v3() -> tuple[
    Callable[[dict[str, Any], EventEmitter], AsyncGenerator[str | None, None]], str
]:
    """Same contract as ``generate_source_list_v2`` but uses inline ``[^ID]``
    citation markers extracted via regex instead of an attribution LLM call.

    The ``llm_provider`` parameter is accepted for interface compatibility
    but is **not used** (no LLM call is made).
    """

    async def _generate_source_list_v3(
        context: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str | None, None]:
        if context.get("_workflow_done"):
            return

        answer_body: str = context.get("answer_body", "")
        # Prefer raw body (with [^ID] markers) for citation extraction;
        # fall back to clean answer_body if no raw body was stored.
        raw_answer_body: str = context.get("_raw_answer_body", answer_body)
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

        yield emitter.tool_call(
            name="citation_extraction",
            input_data={
                "num_sources": len(retrieved_docs),
                "source_ids": [doc.id for doc in retrieved_docs],
                "method": "regex [^ID]",
            },
            description="Udtrækker [^ID]-citationer fra svaret...",
        )

        # Extract cited sources from inline [^ID] markers (no LLM call)
        used_docs = extract_cited_sources(
            response=raw_answer_body,
            retrieved_docs=retrieved_docs,
        )

        yield emitter.tool_result(
            name="citation_extraction",
            result_data={
                "used_ids": [doc.id for doc in used_docs],
                "num_used": len(used_docs),
                "total_retrieved": len(retrieved_docs),
                "fallback_used": "[^" not in raw_answer_body,  # True if no markers were found
            },
        )

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

    return _generate_source_list_v3, "Skriver kildeliste"


def _compose_final_response(
    interpretation: str,
    lead_paragraph: str,
    answer_body: str,
    definitions: list,
) -> str:
    """Compose the final structured response for conversation history.

    (Duplicated from ``generate_source_list._compose_final_response`` and
    ``generate_source_list_v2._compose_final_response`` to keep this module
    self-contained; the originals are unchanged.)
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
