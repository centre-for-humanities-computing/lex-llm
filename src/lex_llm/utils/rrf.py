"""Shared retrieval utilities used across workflow tools."""

from ..api.connectors.lex_db_connector import LexChunk


def reciprocal_rank_fusion(
    *result_lists: list[LexChunk],
    k: int = 60,
) -> list[LexChunk]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Fusion operates at the chunk level. Each unique (article_id, chunk_seq)
    pair gets an RRF score. If the same chunk appears in multiple result lists,
    its scores are summed.

    Args:
        *result_lists: Any number of ranked LexChunk lists (e.g. per-query semantic
            results, per-query FTS results, etc.).
        k: RRF constant (default 60). Higher k dampens the effect of individual ranks.

    Returns:
        Deduplicated, fused list of LexChunks ordered by RRF score.
    """
    rrf_scores: dict[tuple[int, int], float] = {}
    chunk_map: dict[tuple[int, int], LexChunk] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results):
            key = (chunk.article_id, chunk.chunk_seq)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            chunk_map[key] = chunk

    # Sort by RRF score descending
    sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    return [chunk_map[key] for key in sorted_keys]
