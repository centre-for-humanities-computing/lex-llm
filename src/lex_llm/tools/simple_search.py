"""Simple hybrid search tool — raw user query, no expansion.

This is the simplest search strategy: the user's raw query is used
directly for both semantic (vector) and full-text search, then fused
with Reciprocal Rank Fusion.  No LLM calls are made for query
expansion — the tool only calls the LexDB search API.

Results are deduplicated at the article level with the most relevant
chunk as a highlight.
"""

from collections import OrderedDict
from collections.abc import AsyncGenerator, Callable
from typing import Any

from ..api.event_emitter import EventEmitter
from ..api.connectors.lex_db_connector import (
    LexDBConnector,
    LexChunk,
    group_chunks_to_articles,
)
from ..api.event_models import Source
from .search_with_expansion import (
    _reciprocal_rank_fusion,
    _deduplicate_chunks_to_sources,
    _build_search_result,
)


def simple_search(
    index_name: str = "article_embeddings_e5",
    top_k: int = 10,
    top_k_semantic: int = 50,
    top_k_fts: int = 50,
    rrf_k: int = 60,
) -> Callable[[dict[str, Any], EventEmitter], AsyncGenerator[str | None, None]]:
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

    async def _simple_search(
        context: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str | None, None]:
        # Skip if workflow is done
        if context.get("_workflow_done"):
            return

        user_input: str = context.get("user_input", "")
        connector = LexDBConnector()

        # ------------------------------------------------------------------ #
        # Hybrid search — raw user query for both semantic and FTS           #
        # ------------------------------------------------------------------ #
        yield emitter.tool_call(
            name="simple_search",
            input_data={
                "semantic_query": user_input,
                "keyword_query": user_input,
            },
        )

        semantic_chunks = await connector.batch_vector_search(
            queries=[user_input],
            top_k=top_k_semantic,
            index_name=index_name,
        )
        fts_chunks = await connector.batch_fulltext_search(
            queries=[user_input],
            top_k=top_k_fts,
            index_name=index_name,
        )
        fused_chunks = _reciprocal_rank_fusion(
            semantic_results=semantic_chunks,
            fts_results=fts_chunks,
            k=rrf_k,
        )[:top_k]

        yield emitter.tool_result(
            name="simple_search",
            result_data=_build_search_result(
                semantic_chunks, fts_chunks, fused_chunks, rrf_k
            ),
        )

        # ------------------------------------------------------------------ #
        # Deduplicate and write results to context                           #
        # ------------------------------------------------------------------ #
        sources = _deduplicate_chunks_to_sources(fused_chunks)

        context["retrieved_chunks"] = fused_chunks
        context["retrieved_docs"] = group_chunks_to_articles(fused_chunks)
        context["search_results"] = sources

        # Emit the deduplicated source list as a stream event
        yield emitter.sources(sources)

    return _simple_search
