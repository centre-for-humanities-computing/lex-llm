"""Fast two-stage retrieval cascade with merged eval+expand.

Drops HyDE and the advanced expansion stage entirely. Reuses keywords and
subqueries from ``interpret_and_route`` for stage 1, then uses a single
merged evaluation+expansion LLM call between stages.

Stages:
1. simple_retrieval — hybrid search using keywords/subqueries from interpretation
2. corrective intermediate — only when stage 1 fails; driven by expansion
   queries returned from the merged eval+expand call

A cumulative chunk pool across all stages is RRF-fused at the end, so chunks
appearing in both stages get reinforced.
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
from ..prompts_search_synthesis import get_evaluate_and_expand_prompt
from ..utils.rrf import reciprocal_rank_fusion
from ..utils.retrieval_helpers import build_retrieval_result
from .llm_json import parse_json_response
from .retrieval_cascade import (
    _format_docs,
    _run_relevance_evaluation,
    _set_context_success,
)


def retrieval_cascade_fast(
    llm_provider: LLMProvider,
    index_name: str = "article_embeddings_e5",
    top_k: int = 25,
    top_k_semantic: int = 40,
    top_k_fts: int = 40,
    rrf_k: int = 60,
) -> Callable[[dict[str, Any], EventEmitter], AsyncGenerator[str | None, None]]:
    """Creates a fast two-stage retrieval cascade step.

    Uses keywords and subqueries from ``interpret_and_route`` (set earlier
    in the workflow) for the initial search. A single merged eval+expand
    LLM call decides relevance and, when needed, emits corrective queries
    for a second retrieval pass.

    Cumulative raw result lists from both stages are RRF-fused so chunks
    appearing in both get reinforced.

    Sets context keys:
        - retrieved_chunks: list[LexChunk]
        - retrieved_docs: list[LexArticle]
        - insufficient_context: bool
        - insufficient_context_reason: str (if applicable)
    """

    async def _retrieval_cascade_fast(
        context: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str | None, None]:
        if context.get("_workflow_done"):
            return

        user_input: str = context.get("user_input", "")
        interpretation: str = context.get("query_interpretation", user_input)
        keywords: list[str] = context.get("keywords", [user_input])
        queries: list[str] = context.get("subqueries", [user_input])

        connector = LexDBConnector()
        # Cumulative pool of raw ranked result lists for RRF
        chunk_pool: list[list[LexChunk]] = []

        # ---------------------------------------------------------------- #
        # Stage 1 — simple_retrieval                                        #
        # ---------------------------------------------------------------- #
        yield emitter.tool_call(
            name="simple_retrieval",
            input_data={
                "semantic_queries": queries,
                "keyword_queries": keywords,
            },
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

        # Add raw ranked lists to cumulative pool
        for slist in semantic_chunks:
            chunk_pool.append(slist)
        for slist in fts_chunks:
            chunk_pool.append(slist)

        stage1_fused = reciprocal_rank_fusion(*chunk_pool, k=rrf_k)[:top_k]

        yield emitter.tool_result(
            name="simple_retrieval",
            result_data=build_retrieval_result(
                [c for qs in semantic_chunks for c in qs],
                [c for qs in fts_chunks for c in qs],
                stage1_fused,
                rrf_k,
            ),
        )

        # ---------------------------------------------------------------- #
        # Merged eval+expand (one LLM call)                                 #
        # ---------------------------------------------------------------- #
        semantic_queries: list[str] = []
        keyword_queries: list[str] = []
        is_relevant: bool = False
        reason: str = ""

        if not stage1_fused:
            is_relevant = False
            reason = "Ingen søgeresultater fundet"
        else:
            docs = _format_docs(stage1_fused)

            yield emitter.tool_call(
                name="evaluate_and_expand",
                input_data={
                    "user_input": user_input,
                    "interpretation": interpretation,
                    "retrieved_docs": docs,
                },
            )

            messages = get_evaluate_and_expand_prompt(
                user_input=user_input,
                interpretation=interpretation,
                retrieved_docs=docs,
            )
            llm_messages = [
                ConversationMessage(role=m["role"], content=m["content"])  # type: ignore
                for m in messages
            ]
            eval_response = await llm_provider.generate(llm_messages)

            try:
                result = parse_json_response(eval_response)
                is_relevant = bool(result.get("is_relevant", False))
                reason = str(result.get("reason", ""))
                semantic_queries = list(result.get("semantic_queries", []))
                keyword_queries = list(result.get("keyword_queries", []))
            except (ValueError, TypeError):
                # Fallback: assume relevant to avoid blocking
                is_relevant = True
                reason = ""
                semantic_queries = []
                keyword_queries = []

            yield emitter.tool_result(
                name="evaluate_and_expand",
                result_data={
                    "is_relevant": is_relevant,
                    "reason": reason,
                    "semantic_queries": semantic_queries,
                    "keyword_queries": keyword_queries,
                },
            )

        if is_relevant:
            _set_context_success(context, stage1_fused)
            return

        # ---------------------------------------------------------------- #
        # Stage 2 — corrective intermediate retrieval                       #
        # ---------------------------------------------------------------- #
        if not semantic_queries:
            semantic_queries = [interpretation]
        if not keyword_queries:
            keyword_queries = [user_input]

        yield emitter.tool_call(
            name="intermediate_retrieval",
            input_data={
                "semantic_queries": semantic_queries,
                "keyword_queries": keyword_queries,
                "relevance_feedback": reason,
            },
        )

        semantic_chunks = await connector.batch_vector_search(
            queries=[(q, TextType.QUERY) for q in semantic_queries],
            top_k=top_k_semantic,
            index_name=index_name,
        )
        fts_chunks = await connector.batch_fulltext_search(
            queries=keyword_queries,
            top_k=top_k_fts,
            index_name=index_name,
        )

        for slist in semantic_chunks:
            chunk_pool.append(slist)
        for slist in fts_chunks:
            chunk_pool.append(slist)

        fused_chunks = reciprocal_rank_fusion(*chunk_pool, k=rrf_k)[:top_k]

        yield emitter.tool_result(
            name="intermediate_retrieval",
            result_data=build_retrieval_result(
                [c for qs in semantic_chunks for c in qs],
                [c for qs in fts_chunks for c in qs],
                fused_chunks,
                rrf_k,
            ),
        )

        # ---------------------------------------------------------------- #
        # Final relevance evaluation (terminal — no further expansion)      #
        # ---------------------------------------------------------------- #
        async for result in _run_relevance_evaluation(  # type: ignore
            llm_provider=llm_provider,
            user_input=user_input,
            interpretation=interpretation,
            fused_chunks=fused_chunks,
            emitter=emitter,
        ):
            if isinstance(result, dict):
                is_relevant = result["is_relevant"]
                reason = result["reason"]
            else:
                yield result

        if is_relevant:
            _set_context_success(context, fused_chunks)
            return

        # --- All stages exhausted ---
        context["retrieved_chunks"] = fused_chunks
        context["retrieved_docs"] = group_chunks_to_articles(fused_chunks)
        context["insufficient_context"] = True
        context["insufficient_context_reason"] = (
            reason
            or "Søgningen fandt ikke tilstrækkeligt relevante artikler efter to forsøg"
        )

    return _retrieval_cascade_fast
