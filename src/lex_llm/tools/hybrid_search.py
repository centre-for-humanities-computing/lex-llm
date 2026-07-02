"""Simple hybrid search tool — raw user query, no expansion.

This is the simplest search strategy: the user's raw query is used
directly for both semantic (vector) and full-text search, then fused
with Reciprocal Rank Fusion.  No LLM calls are made for query
expansion — the tool only calls the LexDB search API.

Results are deduplicated at the article level with the most relevant
chunk as a highlight.
"""

from collections.abc import AsyncGenerator, Callable
from typing import Any

from lex_db_api.models.text_type import TextType

from ..api.event_emitter import EventEmitter
from ..api.connectors.lex_db_connector import (
    LexDBConnector,
    group_chunks_to_articles,
)
from ..utils.rrf import reciprocal_rank_fusion
from ..utils.retrieval_helpers import (
    build_search_result,
    deduplicate_chunks_to_sources,
)
from ..utils.descriptions import build_search_description


def hybrid_search(
    index_name: str = "article_embeddings_e5",
    top_k: int = 10,
    top_k_semantic: int = 50,
    top_k_fts: int = 50,
    rrf_k: int = 60,
    output_sources: bool = False,
) -> tuple[
    Callable[[dict[str, Any], EventEmitter], AsyncGenerator[str | None, None]], str
]:
    """Creates a simple search step with no query expansion.

    The step:
    1. Uses the raw user query for both semantic and full-text search.
    2. Fuses results with Reciprocal Rank Fusion.
    3. Deduplicates at the article level with the best chunk as highlight.

    No LLM calls are made — only the LexDB search API is contacted.

    Sets context keys:
        - retrieved_chunks: list[LexChunk] — fused chunks, sorted by RRF score
        - retrieved_docs: list[LexArticle] — chunks grouped into articles
        - search_results: list[Source] — deduplicated article-level results
    """

    async def _hybrid_search(
        context: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str | None, None]:
        # Skip if workflow is done
        if context.get("_workflow_done"):
            return

        user_input: str = context.get("user_input", "")
        keywords: list[str] = context.get("keywords", [user_input])
        queries: list[str] = context.get("subqueries", [user_input])
        connector = LexDBConnector()

        # ------------------------------------------------------------------ #
        # Hybrid search — raw user query for both semantic and FTS           #
        # ------------------------------------------------------------------ #
        yield emitter.tool_call(
            name="hybrid_search",
            input_data={
                "semantic_query": queries,
                "keyword_query": keywords,
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
            name="hybrid_search",
            result_data=build_search_result(
                [c for qs in semantic_chunks for c in qs],
                [c for qs in fts_chunks for c in qs],
                fused_chunks,
                rrf_k,
            ),
        )

        # ------------------------------------------------------------------ #
        # Deduplicate and write results to context                           #
        # ------------------------------------------------------------------ #
        sources = deduplicate_chunks_to_sources(fused_chunks)

        context["retrieved_chunks"] = fused_chunks
        context["retrieved_docs"] = group_chunks_to_articles(fused_chunks)
        context["search_results"] = sources

        # Emit the deduplicated source list as a stream event
        if output_sources:
            yield emitter.sources(sources)

    return _hybrid_search, "Søger blandt Lex's artikler..."
