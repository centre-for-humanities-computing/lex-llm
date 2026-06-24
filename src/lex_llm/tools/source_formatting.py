"""Shared helpers for formatting retrieved sources into LLM prompts.

Used by generation steps to inject retrieved sources into the user message
rather than rewriting the system prompt. This keeps the system prompt stable
for KV-cache reuse and follows the standard RAG pattern.
"""

from ..api.connectors.lex_db_connector import (
    LexArticle,
    LexChunk,
    group_chunks_to_articles,
)

# Record format used consistently across all prompts.
# Each source is rendered as:
#   Titel: <title>
#   Indhold: <text>
#   URL: <url>
#   ID: <id>
_SOURCE_TEMPLATE = "Titel: {title}\nIndhold: {text}\nURL: {url}\nID: {id}"


def format_sources_block(articles: list[LexArticle]) -> str:
    """Format a list of LexArticle objects into a sources text block.

    Articles are rendered in the order given (caller is responsible for
    sorting by (article_id, chunk_seq) before grouping if using chunks).
    """
    return "\n\n".join(
        _SOURCE_TEMPLATE.format(
            title=doc.title,
            text=doc.text,
            url=doc.url or "",
            id=doc.id,
        )
        for doc in articles
    )


def build_user_message_with_sources(
    user_input: str,
    retrieved_chunks: list[LexChunk] | None = None,
    retrieved_docs: list[LexArticle] | None = None,
) -> str:
    """Build a user message with retrieved sources appended.

    Sources are sorted by (article_id, chunk_seq) and grouped into articles
    to maximize KV-cache hits during generation. The sources block is
    appended under a ``# Kilder`` heading.

    Args:
        user_input: The clean user query.
        retrieved_chunks: Raw chunks from retrieval (sorted + grouped).
        retrieved_docs: Pre-grouped articles (used as-is).

    Returns:
        The user message with sources appended, or just user_input if
        no sources are provided.
    """
    if retrieved_chunks:
        sorted_chunks = sorted(
            retrieved_chunks, key=lambda c: (c.article_id, c.chunk_seq)
        )
        articles = group_chunks_to_articles(sorted_chunks)
    elif retrieved_docs:
        articles = retrieved_docs
    else:
        return user_input

    if not articles:
        return user_input

    sources_block = format_sources_block(articles)
    return f"{user_input}\n\n# Kilder\n{sources_block}"
