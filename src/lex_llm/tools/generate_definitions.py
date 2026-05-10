"""Definitions generation step."""

from collections.abc import AsyncGenerator, Callable
from typing import Any

from ..api.event_emitter import EventEmitter
from ..api.connectors.openai_provider import LLMProvider
from ..api.event_models import ConversationMessage, DefinitionItem
from ..prompts_search_synthesis import get_definitions_prompt
from .llm_json import parse_json_response


def generate_definitions(
    llm_provider: LLMProvider,
) -> Callable[[dict[str, Any], EventEmitter], AsyncGenerator[str | None, None]]:
    """Creates a step that extracts key term definitions from the answer body.

    Uses structured JSON output to return a list of term-definition pairs.

    Sets context keys:
        - definitions: list[DefinitionItem] — the extracted definitions
    """

    async def _generate_definitions(
        context: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str | None, None]:
        # Skip if workflow is done
        if context.get("_workflow_done"):
            return

        answer_body: str = context.get("answer_body", "")

        if not answer_body:
            return

        messages = get_definitions_prompt(answer_body=answer_body)

        llm_messages = [
            ConversationMessage(role=m["role"], content=m["content"])  # type: ignore
            for m in messages
        ]

        raw_response = await llm_provider.generate(llm_messages)

        try:
            result = parse_json_response(raw_response)
            raw_definitions = result.get("definitions", [])
            definitions: list[DefinitionItem] = []
            for item in raw_definitions:
                term = item.get("term", "").strip()
                definition = item.get("definition", "").strip()
                if term and definition:
                    definitions.append(DefinitionItem(term=term, definition=definition))
        except ValueError:
            # Fallback: no definitions if parsing fails
            definitions = []

        context["definitions"] = definitions

        if definitions:
            yield emitter.definitions(definitions)

    return _generate_definitions
