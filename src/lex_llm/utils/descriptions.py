"""Shared utilities for building user-facing descriptions."""


def build_search_description(
    keywords: list[str] | None = None,
    queries: list[str] | None = None,
) -> str:
    """Build a Danish user-facing description of what is being searched for.

    Format:
        Søger med nøgleord...
        keyword1
        keyword2

        Søger med underspørgsmål...
        query1
        query2

    Args:
        keywords: Keyword terms used for full-text search.
        queries: Semantic subqueries used for vector search.

    Returns:
        A human-readable description string, or empty string if both are empty.
    """
    keywords = keywords or []
    queries = queries or []

    if not keywords and not queries:
        return ""

    parts: list[str] = []
    if keywords:
        parts.append("Søger med nøgleord...\n" + "\n".join(keywords))
    if queries:
        parts.append("Søger med underspørgsmål...\n" + "\n".join(queries))

    return "\n\n".join(parts)
