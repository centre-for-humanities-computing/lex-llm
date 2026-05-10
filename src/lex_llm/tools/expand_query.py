"""Query expansion step using HyDE and keyword generation."""

from collections.abc import AsyncGenerator, Callable
from typing import Any

from ..api.event_emitter import EventEmitter
from ..api.connectors.openai_provider import LLMProvider
from ..api.event_models import ConversationMessage
from ..prompts_search_synthesis import (
    get_hyde_prompt,
    get_keyword_expansion_prompt,
)
from .llm_json import parse_json_response


def expand_query(
    llm_provider: LLMProvider,
) -> Callable[[dict[str, Any], EventEmitter], AsyncGenerator[str | None, None]]:
    """Creates a step that expands the query using HyDE and keyword generation.

    Generates:
    1. A set of HyDE passages for semantic search
    2. A set of keyword search queries for full-text search

    Sets context keys:
        - hyde_passages: list[str] — individual HyDE passages
        - keyword_queries: list[str] — keyword queries for full-text search
    """

    async def _expand_query(
        context: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str | None, None]:
        # Skip if workflow is done (e.g., out-of-scope query)
        if context.get("_workflow_done"):
            return

        user_input: str = context.get("user_input", "")
        interpretation: str = context.get("query_interpretation", user_input)

        # Emit tool_call event for observability        
        yield emitter.tool_call(
            name="expand_query",
            input_data={
                "user_input": user_input,
                "query_interpretation": interpretation,
            },
        )
        
        # --- Generate HyDE passages ---
        hyde_messages = get_hyde_prompt(
            user_input=user_input,
            interpretation=interpretation,
        )
        llm_hyde_messages = [
            ConversationMessage(role=m["role"], content=m["content"])  # type: ignore
            for m in hyde_messages
        ]
        hyde_response = await llm_provider.generate(llm_hyde_messages)
        hyde_response = hyde_response.strip()

        # Parse HyDE response — may be JSON with passages or plain text
        try:
            hyde_result = parse_json_response(hyde_response)
            hyde_passages: list[str] = hyde_result.get("passages", [hyde_response])
        except ValueError:
            # Fallback: treat the whole response as a single passage
            hyde_passages = [hyde_response]

        context["hyde_passages"] = hyde_passages

        # --- Generate keyword queries ---
        keyword_messages = get_keyword_expansion_prompt(
            user_input=user_input,
            interpretation=interpretation,
        )
        llm_keyword_messages = [
            ConversationMessage(role=m["role"], content=m["content"])  # type: ignore
            for m in keyword_messages
        ]
        keyword_response = await llm_provider.generate(llm_keyword_messages)

        try:
            keyword_result = parse_json_response(keyword_response)
            keyword_queries: list[str] = keyword_result.get("queries", [user_input])
        except ValueError:
            # Fallback: use the user input as a single keyword query
            keyword_queries = [user_input]

        context["keyword_queries"] = keyword_queries

        # Emit tool_result event for observability
        yield emitter.tool_result(
            name="expand_query",
            result_data={
                "hyde_documents": hyde_passages,
                "keyword_queries": keyword_queries,
            },
        )

    return _expand_query
