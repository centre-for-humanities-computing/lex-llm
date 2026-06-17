"""Search and validate step with a three-stage progressive retrieval strategy.

The three stages escalate in compute cost:
1. simple_retrieval     — raw user query, no expansion
2. intermediate_retrieval — short semantic subqueries + expanded keywords (single LLM call)
3. advanced_retrieval   — HyDE passages + broadened keywords (single LLM call)

Each stage evaluates relevance before escalating to the next.
"""

from collections.abc import AsyncGenerator, Callable
from typing import Any

from lex_db_api.models.text_type import TextType

from ..api.event_emitter import EventEmitter
from ..api.connectors.openai_provider import LLMProvider
from ..api.connectors.lex_db_connector import (
    LexDBConnector,
    LexChunk,
    group_chunks_to_articles,
)
from ..api.event_models import ConversationMessage
from ..prompts_search_synthesis import (
    get_relevance_evaluation_prompt,
    get_intermediate_expansion_prompt,
    get_advanced_expansion_prompt,
)
from ..utils.rrf import reciprocal_rank_fusion
from ..utils.retrieval_helpers import build_retrieval_result
from ..utils.descriptions import build_search_description
from .llm_json import parse_json_response


def _format_docs(chunks: list[LexChunk]) -> str:
    """Format retrieved chunks as a summary for the LLM.

    Chunks are grouped by article_id and each article gets its ID, title,
    and a truncated excerpt from the combined chunk text.
    """
    articles = group_chunks_to_articles(chunks)
    lines = []
    for doc in articles:
        lines.append(f"*ID:* {doc.id} | *Titel:* {doc.title}\n*Tekst:* {doc.text}\n")
    return "\n\n".join(lines)


def retrieval_cascade(
    llm_provider: LLMProvider,
    index_name: str = "article_embeddings_e5",
    top_k: int = 10,
    top_k_semantic: int = 50,
    top_k_fts: int = 50,
    rrf_k: int = 60,
) -> tuple[
    Callable[[dict[str, Any], EventEmitter], AsyncGenerator[str | None, None]], str
]:
    """Creates a step that searches the knowledge base with a three-stage progressive strategy.

    The three stages escalate in compute cost, stopping as soon as a stage
    returns relevant results:

    1. simple_retrieval — hybrid search with the raw user query, no expansion.
    2. intermediate_retrieval — short semantic subqueries + expanded keyword queries,
       both generated in a single LLM call informed by the stage-1 relevance feedback.
    3. advanced_retrieval — HyDE passages + broadened keyword queries, both generated
       in a single LLM call informed by the stage-2 relevance feedback.

    Sets context keys:
        - retrieved_chunks: list[LexChunk] — the final fused chunks, sorted by RRF score
        - retrieved_docs: list[LexArticle] — chunks grouped into articles for downstream
        - insufficient_context: bool — True if all three stages failed to find relevant results
        - insufficient_context_reason: str — reason for insufficient context (if applicable)
    """

    async def _retrieval_cascade(
        context: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str | None, None]:
        # Skip if workflow is done (e.g., out-of-scope query)
        if context.get("_workflow_done"):
            return

        user_input: str = context.get("user_input", "")
        interpretation: str = context.get("query_interpretation", user_input)
        keywords: list[str] = context.get("keywords", [user_input])
        queries: list[str] = context.get("subqueries", [user_input])

        connector = LexDBConnector()
        best_chunks: list[LexChunk] = []
        best_relevance_reason = ""

        # ------------------------------------------------------------------ #
        # Stage 1 — simple_retrieval                                          #
        # Raw user query used directly for both semantic and FTS search.      #
        # ------------------------------------------------------------------ #
        yield emitter.tool_call(
            name="simple_retrieval",
            input_data={
                "semantic_queries": [queries],
                "keyword_queries": [keywords],
            },
            description=build_search_description(
                keywords=keywords,
                queries=queries,
            ),
        )

        semantic_chunks = await connector.batch_vector_search(
            queries=[(q, TextType.QUERY) for q in queries],
            top_k=top_k_semantic,
            index_name=index_name,
        )
        fts_chunks = await connector.batch_fulltext_search(
            queries=keywords,
            top_k=top_k_fts,
            index_name=index_name,
        )
        fused_chunks = reciprocal_rank_fusion(
            *semantic_chunks,
            *fts_chunks,
            k=rrf_k,
        )[:top_k]

        yield emitter.tool_result(
            name="simple_retrieval",
            result_data=build_retrieval_result(
                [c for qs in semantic_chunks for c in qs],
                [c for qs in fts_chunks for c in qs],
                fused_chunks,
                rrf_k,
            ),
        )

        if fused_chunks:
            best_chunks = fused_chunks

        is_relevant: bool = False
        reason: str = ""
        refinement: str = ""
        async for result in _run_relevance_evaluation(
            llm_provider=llm_provider,
            user_input=user_input,
            interpretation=interpretation,
            fused_chunks=fused_chunks,
            emitter=emitter,
        ):
            if isinstance(result, dict):
                is_relevant = result["is_relevant"]
                reason = result["reason"]
                refinement = result["suggested_query_refinement"]
            else:
                yield result

        if is_relevant:
            _set_context_success(context, fused_chunks)
            return

        best_relevance_reason = reason

        # ------------------------------------------------------------------ #
        # Stage 2 — intermediate_retrieval                                    #
        # Short semantic subqueries + expanded keyword queries, both from     #
        # a single LLM call informed by stage-1 relevance feedback.           #
        # ------------------------------------------------------------------ #
        (
            intermediate_semantic_queries,
            expanded_keyword_queries,
        ) = await _intermediate_expansion(
            llm_provider=llm_provider,
            user_input=user_input,
            interpretation=interpretation,
            relevance_feedback=reason,
        )

        yield emitter.tool_call(
            name="intermediate_retrieval",
            input_data={
                "semantic_queries": intermediate_semantic_queries,
                "keyword_queries": expanded_keyword_queries,
                "relevance_feedback": reason,
            },
            description=build_search_description(
                keywords=expanded_keyword_queries,
                queries=intermediate_semantic_queries,
            ),
        )

        semantic_chunks = await connector.batch_vector_search(
            queries=[(q, TextType.QUERY) for q in intermediate_semantic_queries],
            top_k=top_k_semantic,
            index_name=index_name,
        )
        fts_chunks = await connector.batch_fulltext_search(
            queries=expanded_keyword_queries,
            top_k=top_k_fts,
            index_name=index_name,
        )
        fused_chunks = reciprocal_rank_fusion(
            *semantic_chunks,
            *fts_chunks,
            k=rrf_k,
        )[:top_k]

        yield emitter.tool_result(
            name="intermediate_retrieval",
            result_data=build_retrieval_result(
                [c for qs in semantic_chunks for c in qs],
                [c for qs in fts_chunks for c in qs],
                fused_chunks,
                rrf_k,
            ),
        )

        if len(fused_chunks) > len(best_chunks):
            best_chunks = fused_chunks

        async for result in _run_relevance_evaluation(
            llm_provider=llm_provider,
            user_input=user_input,
            interpretation=interpretation,
            fused_chunks=fused_chunks,
            emitter=emitter,
        ):
            if isinstance(result, dict):
                is_relevant = result["is_relevant"]
                reason = result["reason"]
                refinement = result["suggested_query_refinement"]
            else:
                yield result

        if is_relevant:
            _set_context_success(context, fused_chunks)
            return

        best_relevance_reason = reason

        # ------------------------------------------------------------------ #
        # Stage 3 — advanced_retrieval                                        #
        # HyDE passages + broadened keyword queries, both from a single LLM   #
        # call informed by stage-2 relevance feedback.                        #
        # ------------------------------------------------------------------ #
        hyde_passages, broadened_keyword_queries = await _advanced_expansion(
            llm_provider=llm_provider,
            user_input=user_input,
            interpretation=interpretation,
            previous_semantic_queries=intermediate_semantic_queries,
            previous_keyword_queries=expanded_keyword_queries,
            refinement_suggestion=refinement or reason,
        )

        yield emitter.tool_call(
            name="advanced_retrieval",
            input_data={
                "semantic_queries": hyde_passages,
                "keyword_queries": broadened_keyword_queries,
                "relevance_feedback": reason,
            },
            description=build_search_description(
                keywords=broadened_keyword_queries,
                queries=hyde_passages,
            ),
        )

        semantic_chunks = await connector.batch_vector_search(
            queries=[(p, TextType.PASSAGE) for p in hyde_passages],
            top_k=top_k_semantic,
            index_name=index_name,
        )
        fts_chunks = await connector.batch_fulltext_search(
            queries=broadened_keyword_queries,
            top_k=top_k_fts,
            index_name=index_name,
        )
        fused_chunks = reciprocal_rank_fusion(
            *semantic_chunks,
            *fts_chunks,
            k=rrf_k,
        )[:top_k]

        yield emitter.tool_result(
            name="advanced_retrieval",
            result_data=build_retrieval_result(
                [c for qs in semantic_chunks for c in qs],
                [c for qs in fts_chunks for c in qs],
                fused_chunks,
                rrf_k,
            ),
        )

        if len(fused_chunks) > len(best_chunks):
            best_chunks = fused_chunks

        async for result in _run_relevance_evaluation(
            llm_provider=llm_provider,
            user_input=user_input,
            interpretation=interpretation,
            fused_chunks=fused_chunks,
            emitter=emitter,
        ):
            if isinstance(result, dict):
                is_relevant = result["is_relevant"]
                reason = result["reason"]
                refinement = result["suggested_query_refinement"]
            else:
                yield result

        if is_relevant:
            _set_context_success(context, fused_chunks)
            return

        best_relevance_reason = reason or best_relevance_reason

        # --- All stages exhausted ---
        context["retrieved_chunks"] = best_chunks
        context["retrieved_docs"] = group_chunks_to_articles(best_chunks)
        context["insufficient_context"] = True
        context["insufficient_context_reason"] = (
            best_relevance_reason
            or "Søgningen fandt ikke tilstrækkeligt relevante artikler efter tre forsøg"
        )

    return _retrieval_cascade, "Søger blandt Lex's artikler"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _set_context_success(context: dict[str, Any], chunks: list[LexChunk]) -> None:
    """Write successful retrieval results into the workflow context."""
    context["retrieved_chunks"] = chunks
    context["retrieved_docs"] = group_chunks_to_articles(chunks)
    context["insufficient_context"] = False


async def _run_relevance_evaluation(
    llm_provider: LLMProvider,
    user_input: str,
    interpretation: str,
    fused_chunks: list[LexChunk],
    emitter: EventEmitter,
) -> AsyncGenerator[dict[str, Any] | str, None]:
    """Call the LLM to evaluate relevance and yield events in real-time.

    Yields SSE event strings for the ``relevance_evaluation`` tool_call (before
    the LLM call) and tool_result (after), so callers see the tool call event
    immediately. The final yield is the ``result_data`` dict containing
    ``is_relevant``, ``reason``, and ``suggested_query_refinement``.

    Yields a result dict with ``is_relevant=False`` when there are no chunks
    to avoid infinite escalation.
    """
    if not fused_chunks:
        yield {
            "is_relevant": False,
            "reason": "Ingen søgeresultater fundet",
            "suggested_query_refinement": "",
        }
        return

    docs = _format_docs(fused_chunks)

    yield emitter.tool_call(
        name="relevance_evaluation",
        input_data={
            "user_input": user_input,
            "interpretation": interpretation,
            "retrieved_docs": docs,
        },
        description="Vurderer om resultaterne er relevante",
    )

    eval_messages = get_relevance_evaluation_prompt(
        user_input=user_input,
        interpretation=interpretation,
        retrieved_docs=docs,
    )
    llm_eval_messages = [
        ConversationMessage(role=m["role"], content=m["content"])  # type: ignore
        for m in eval_messages
    ]
    eval_response = await llm_provider.generate(llm_eval_messages)

    try:
        eval_result = parse_json_response(eval_response)
        is_relevant: bool = eval_result.get("is_relevant", False)
        reason: str = eval_result.get("reason", "")
        refinement: str = eval_result.get("suggested_query_refinement", "")
    except ValueError:
        # If we can't parse, assume relevant to avoid blocking the workflow
        is_relevant = True
        reason = ""
        refinement = ""

    result_data = {
        "is_relevant": is_relevant,
        "reason": reason,
        "suggested_query_refinement": refinement,
    }

    yield emitter.tool_result(
        name="relevance_evaluation",
        result_data=result_data,
    )

    yield result_data


async def _intermediate_expansion(
    llm_provider: LLMProvider,
    user_input: str,
    interpretation: str,
    relevance_feedback: str,
) -> tuple[list[str], list[str]]:
    """Single LLM call that returns (semantic_subqueries, keyword_queries) for stage 2.

    Semantic subqueries are short phrases/sentences — faster to embed than HyDE passages.
    """
    messages = get_intermediate_expansion_prompt(
        user_input=user_input,
        interpretation=interpretation,
        relevance_feedback=relevance_feedback,
    )
    llm_messages = [
        ConversationMessage(role=m["role"], content=m["content"])  # type: ignore
        for m in messages
    ]
    response = await llm_provider.generate(llm_messages)
    try:
        result = parse_json_response(response)
        semantic_queries: list[str] = result.get("semantic_queries", [interpretation])
        keyword_queries: list[str] = result.get("keyword_queries", [user_input])
    except ValueError:
        semantic_queries = [interpretation]
        keyword_queries = [user_input]
    return semantic_queries, keyword_queries


async def _advanced_expansion(
    llm_provider: LLMProvider,
    user_input: str,
    interpretation: str,
    previous_semantic_queries: list[str],
    previous_keyword_queries: list[str],
    refinement_suggestion: str,
) -> tuple[list[str], list[str]]:
    """Single LLM call that returns (hyde_passages, broadened_keyword_queries) for stage 3.

    HyDE passages are longer hypothetical encyclopedia paragraphs used for semantic search.
    """
    messages = get_advanced_expansion_prompt(
        user_input=user_input,
        interpretation=interpretation,
        previous_semantic_queries=previous_semantic_queries,
        previous_keyword_queries=previous_keyword_queries,
        refinement_suggestion=refinement_suggestion,
    )
    llm_messages = [
        ConversationMessage(role=m["role"], content=m["content"])  # type: ignore
        for m in messages
    ]
    response = await llm_provider.generate(llm_messages)
    response = response.strip()
    try:
        result = parse_json_response(response)
        passages: list[str] = result.get("passages", [interpretation])
        keyword_queries: list[str] = result.get(
            "keyword_queries", previous_keyword_queries
        )
    except ValueError:
        passages = [interpretation]
        keyword_queries = previous_keyword_queries
    return passages, keyword_queries
