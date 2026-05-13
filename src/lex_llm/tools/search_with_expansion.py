"""Search-only tool with query expansion and hybrid retrieval.

Unlike ``search_and_validate``, this tool does NOT evaluate relevance
or escalate through multiple stages. It performs a single round of
query expansion (semantic subqueries + keyword queries) followed by
hybrid search with Reciprocal Rank Fusion, and returns deduplicated
article-level results with the most relevant chunk as a highlight.

This is designed for search-endpoint workflows that only need a list
of matching articles — no summarization, no corrective-RAG loops.
"""

from collections import OrderedDict
from collections.abc import AsyncGenerator, Callable
from typing import Any

from ..api.event_emitter import EventEmitter
from ..api.connectors.openai_provider import LLMProvider
from ..api.connectors.lex_db_connector import (
    LexDBConnector,
    LexChunk,
    group_chunks_to_articles,
)
from ..api.event_models import ConversationMessage, Source
from ..prompts_search_synthesis import get_intermediate_expansion_prompt
from .llm_json import parse_json_response


def _reciprocal_rank_fusion(
    semantic_results: list[LexChunk],
    fts_results: list[LexChunk],
    k: int = 60,
) -> list[LexChunk]:
    """Merge semantic and full-text search results using Reciprocal Rank Fusion.

    Fusion operates at the chunk level. Each unique (article_id, chunk_seq)
    pair gets an RRF score. If the same chunk appears in both result sets,
    its scores are summed.

    Args:
        semantic_results: Chunks from vector/semantic search, ordered by relevance.
        fts_results: Chunks from full-text search, ordered by relevance.
        k: RRF constant (default 60). Higher k dampens the effect of individual ranks.

    Returns:
        Deduplicated, fused list of LexChunks ordered by RRF score.
    """
    rrf_scores: dict[tuple[int, int], float] = {}
    chunk_map: dict[tuple[int, int], LexChunk] = {}

    for rank, chunk in enumerate(semantic_results):
        key = (chunk.article_id, chunk.chunk_seq)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[key] = chunk

    for rank, chunk in enumerate(fts_results):
        key = (chunk.article_id, chunk.chunk_seq)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[key] = chunk

    sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    return [chunk_map[key] for key in sorted_keys]


def search_with_expansion(
    llm_provider: LLMProvider,
    index_name: str = "article_embeddings_e5",
    top_k: int = 10,
    top_k_semantic: int = 50,
    top_k_fts: int = 50,
    rrf_k: int = 60,
) -> Callable[[dict[str, Any], EventEmitter], AsyncGenerator[str | None, None]]:
    """Creates a search step that expands the query and performs hybrid retrieval.

    The step:
    1. Calls the LLM to generate semantic subqueries + keyword queries
       (using the intermediate expansion prompt).
    2. Runs batch vector search with the semantic subqueries.
    3. Runs batch full-text search with the keyword queries.
    4. Fuses results with Reciprocal Rank Fusion.
    5. Returns the fused chunks grouped as articles.

    Sets context keys:
        - retrieved_chunks: list[LexChunk] — fused chunks, sorted by RRF score
        - retrieved_docs: list[LexArticle] — chunks grouped into articles
        - search_queries: dict — the expanded queries used for the search
    """

    async def _search_with_expansion(
        context: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str | None, None]:
        # Skip if workflow is done (e.g., out-of-scope query)
        if context.get("_workflow_done"):
            return

        user_input: str = context.get("user_input", "")
        interpretation: str = "Brugerens forespørgsel er en søgning efter artikler relateret til det indtastede." 

        connector = LexDBConnector()

        # ------------------------------------------------------------------ #
        # Step 1 — Query expansion                                           #
        # Generate semantic subqueries + keyword queries via LLM.            #
        # ------------------------------------------------------------------ #
        yield emitter.tool_call(
            name="query_expansion",
            input_data={
                "user_input": user_input,
                "interpretation": interpretation,
            },
        )

        semantic_queries, keyword_queries = await _expand_queries(
            llm_provider=llm_provider,
            user_input=user_input,
            interpretation=interpretation,
        )

        yield emitter.tool_result(
            name="query_expansion",
            result_data={
                "semantic_queries": semantic_queries,
                "keyword_queries": keyword_queries,
            },
        )

        # ------------------------------------------------------------------ #
        # Step 2 — Hybrid search                                             #
        # Batch vector search + batch FTS + RRF fusion.                      #
        # ------------------------------------------------------------------ #
        yield emitter.tool_call(
            name="hybrid_search",
            input_data={
                "semantic_queries": semantic_queries,
                "keyword_queries": keyword_queries,
            },
        )

        semantic_chunks = await connector.batch_vector_search(
            queries=semantic_queries,
            top_k=top_k_semantic,
            index_name=index_name,
        )
        fts_chunks = await connector.batch_fulltext_search(
            queries=keyword_queries,
            top_k=top_k_fts,
            index_name=index_name,
        )
        fused_chunks = _reciprocal_rank_fusion(
            semantic_results=semantic_chunks,
            fts_results=fts_chunks,
            k=rrf_k,
        )[:top_k]

        yield emitter.tool_result(
            name="hybrid_search",
            result_data=_build_search_result(
                semantic_chunks, fts_chunks, fused_chunks, rrf_k
            ),
        )

        # ------------------------------------------------------------------ #
        # Step 3 — Deduplicate and write results to context                  #
        # Group by article_id, keep the highest-ranked chunk as highlight.   #
        # ------------------------------------------------------------------ #
        sources = _deduplicate_chunks_to_sources(fused_chunks)

        context["retrieved_chunks"] = fused_chunks
        context["retrieved_docs"] = group_chunks_to_articles(fused_chunks)
        context["search_queries"] = {
            "semantic_queries": semantic_queries,
            "keyword_queries": keyword_queries,
        }
        context["search_results"] = sources

        # Emit the deduplicated source list as a stream event
        yield emitter.sources(sources)

    return _search_with_expansion


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _expand_queries(
    llm_provider: LLMProvider,
    user_input: str,
    interpretation: str,
) -> tuple[list[str], list[str]]:
    """Call the LLM to generate semantic subqueries and keyword queries.

    Uses the intermediate expansion prompt which produces short semantic
    subqueries (for vector search) and keyword queries (for full-text search)
    in a single LLM call.
    """
    messages = get_intermediate_expansion_prompt(
        user_input=user_input,
        interpretation=interpretation,
        relevance_feedback="Første søgning — udvid forespørgslen for at finde relevante artikler",
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


def _deduplicate_chunks_to_sources(chunks: list[LexChunk]) -> list[Source]:
    """Deduplicate chunks by article_id, keeping the best chunk as a highlight.

    Because ``chunks`` is already sorted by RRF score (descending), the
    first chunk encountered for each article is guaranteed to be the most
    relevant one.  An ``OrderedDict`` preserves insertion order so the
    output list retains the original ranking.
    """
    best: OrderedDict[int, LexChunk] = OrderedDict()
    for chunk in chunks:
        if chunk.article_id not in best:
            best[chunk.article_id] = chunk

    return [
        Source(
            id=chunk.article_id,
            title=chunk.title or "",
            url=chunk.url,
            highlight=chunk.chunk_text,
        )
        for chunk in best.values()
    ]


def _build_search_result(
    semantic_chunks: list[LexChunk],
    fts_chunks: list[LexChunk],
    fused_chunks: list[LexChunk],
    rrf_k: int,
) -> dict[str, Any]:
    """Build a serialisable result dict for tool_result events.

    The ``results`` key contains deduplicated article-level entries with
    the most relevant chunk text as ``highlight``.
    """
    sources = _deduplicate_chunks_to_sources(fused_chunks)
    return {
        "semantic_chunks": [
            {
                "article_id": chunk.article_id,
                "chunk_seq": chunk.chunk_seq,
                "title": chunk.title,
                "url": chunk.url,
                "score": round(1.0 / (rrf_k + idx + 1), 4),
            }
            for idx, chunk in enumerate(semantic_chunks)
        ],
        "fts_chunks": [
            {
                "article_id": chunk.article_id,
                "chunk_seq": chunk.chunk_seq,
                "title": chunk.title,
                "url": chunk.url,
                "score": round(1.0 / (rrf_k + idx + 1), 4),
            }
            for idx, chunk in enumerate(fts_chunks)
        ],
        "results": [
            {
                "id": s.id,
                "title": s.title,
                "url": s.url,
                "highlight": s.highlight,
            }
            for s in sources
        ],
    }
