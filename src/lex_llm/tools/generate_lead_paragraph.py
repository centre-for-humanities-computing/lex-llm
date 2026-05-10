"""Lead paragraph generation step."""

from collections.abc import AsyncGenerator, Callable
from typing import Any

from ..api.event_emitter import EventEmitter
from ..api.connectors.openai_provider import LLMProvider
from ..api.event_models import ConversationMessage
from ..prompts_search_synthesis import get_lead_paragraph_prompt


def generate_lead_paragraph(
    llm_provider: LLMProvider,
) -> Callable[[dict[str, Any], EventEmitter], AsyncGenerator[str | None, None]]:
    """Creates a step that generates a lead paragraph from the answer body.

    The lead paragraph brings the conclusion to the front, respecting
    the reader's time.

    Sets context keys:
        - lead_paragraph: str — the generated lead paragraph
    """

    async def _generate_lead_paragraph(
        context: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str | None, None]:
        # Skip if workflow is done
        if context.get("_workflow_done"):
            return

        user_input: str = context.get("user_input", "")
        interpretation: str = context.get("query_interpretation", user_input)
        answer_body: str = context.get("answer_body", "")

        if not answer_body:
            return

        messages = get_lead_paragraph_prompt(
            user_input=user_input,
            interpretation=interpretation,
            answer_body=answer_body,
        )

        llm_messages = [
            ConversationMessage(role=m["role"], content=m["content"])  # type: ignore
            for m in messages
        ]

        # Stream the lead paragraph
        full_paragraph = ""
        async for chunk in llm_provider.generate_stream(llm_messages):  # type: ignore
            full_paragraph += chunk
            yield emitter.lead_paragraph_chunk(chunk)

        context["lead_paragraph"] = full_paragraph

    return _generate_lead_paragraph
