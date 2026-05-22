"""Shared retrieval result helpers used across workflow tools."""

from collections import OrderedDict
from typing import Any

from ..api.connectors.lex_db_connector import LexChunk
from ..api.event_models import Source


def build_retrieval_result(
    semantic_chunks: list[LexChunk],
    fts_chunks: list[LexChunk],
    fused_chunks: list[LexChunk],
    rrf_k: int,
) -> dict[str, Any]:
    """Build the base serialisable result dict for tool_result events.

    Returns a dict with ``semantic_chunks``, ``fts_chunks``, and
    ``top_fused_chunks`` keys.  Callers that need article-level deduplication
    should add a ``results`` key by calling :func:`deduplicate_chunks_to_sources`.
    """
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
        "top_fused_chunks": [
            {
                "article_id": chunk.article_id,
                "chunk_seq": chunk.chunk_seq,
                "title": chunk.title,
                "url": chunk.url,
                "rrf_score": round(1.0 / (rrf_k + idx + 1), 4),
            }
            for idx, chunk in enumerate(fused_chunks)
        ],
    }


def deduplicate_chunks_to_sources(chunks: list[LexChunk]) -> list[Source]:
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


def build_search_result(
    semantic_chunks: list[LexChunk],
    fts_chunks: list[LexChunk],
    fused_chunks: list[LexChunk],
    rrf_k: int,
) -> dict[str, Any]:
    """Build a serialisable result dict with article-level deduplication.

    Like :func:`build_retrieval_result` but adds a ``results`` key containing
    deduplicated article-level entries with the most relevant chunk text as
    ``highlight``.
    """
    base = build_retrieval_result(semantic_chunks, fts_chunks, fused_chunks, rrf_k)
    sources = deduplicate_chunks_to_sources(fused_chunks)
    base["results"] = [
        {
            "id": s.id,
            "title": s.title,
            "url": s.url,
            "highlight": s.highlight,
        }
        for s in sources
    ]
    return base
