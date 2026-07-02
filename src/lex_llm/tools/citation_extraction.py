"""Inline citation extraction — pure regex, no LLM.

Extracts ``[^ID]`` citation markers from generated response text and maps
them back to the corresponding ``LexArticle`` objects. Used by v3 generation
tools to replace the explicit source-attribution LLM call.
"""

import re
from ..api.connectors.lex_db_connector import LexArticle

# Matches markdown-footnote-style citation markers: [^675], [^42], etc.
_CITATION_RE = re.compile(r"\[\^(\d+)\]")

# Number of characters to hold back during streaming to handle partial
# citation markers that may be split across chunk boundaries.  A citation
# like "[^675]" is at most ~8 chars, so 20 is a safe margin.
_STRIP_BUFFER_SIZE = 20


class CitationStripper:
    """Streaming citation stripper that handles chunk-boundary edge cases.

    Typical tokenizers don't split ``[^`` or ``]`` mid-stream, but this
    buffer provides a safety net by holding back a trailing segment of
    each chunk to avoid emitting partial citation fragments.

    Usage::

        stripper = CitationStripper()
        for chunk in llm_stream:
            clean = stripper.feed(chunk)
            if clean:
                yield emitter.text_chunk(clean)
        tail = stripper.flush()
        if tail:
            yield emitter.text_chunk(tail)
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, chunk: str) -> str:
        """Feed a raw chunk; return clean text safe to emit.

        Holds back the last ``_STRIP_BUFFER_SIZE`` characters of the
        cleaned output to guard against incomplete citation markers.
        """
        combined = self._buffer + chunk
        cleaned = _CITATION_RE.sub("", combined)
        if len(cleaned) > _STRIP_BUFFER_SIZE:
            emit = cleaned[:-_STRIP_BUFFER_SIZE]
            self._buffer = cleaned[-_STRIP_BUFFER_SIZE:]
            return emit
        else:
            self._buffer = cleaned
            return ""

    def flush(self) -> str:
        """Return any remaining buffered text (call after last chunk)."""
        result = self._buffer
        self._buffer = ""
        return result


def strip_citations(text: str) -> str:
    """Remove all ``[^ID]`` citation markers from *text*.

    Convenience wrapper for non-streaming use.
    """
    return _CITATION_RE.sub("", text)


def extract_cited_sources(
    response: str,
    retrieved_docs: list[LexArticle],
) -> list[LexArticle]:
    """Extract cited sources from inline ``[^ID]`` markers in the response.

    Scans the response for ``[^<id>]`` patterns, collects unique IDs in
    order of first appearance, and returns the matching ``LexArticle``
    objects. IDs that don't match any retrieved document are silently
    ignored.

    If no valid citation markers are found, falls back to returning all
    ``retrieved_docs`` (the safe default — avoids empty source lists when
    the model forgets to cite).

    Args:
        response: The generated answer text containing ``[^ID]`` markers.
        retrieved_docs: All documents retrieved for this turn.

    Returns:
        ``LexArticle`` objects that were cited, in order of first citation.
    """
    if not response.strip():
        return []

    # Collect unique IDs in order of first appearance
    seen: set[int] = set()
    cited_ids: list[int] = []
    for match in _CITATION_RE.finditer(response):
        doc_id = int(match.group(1))
        if doc_id not in seen:
            seen.add(doc_id)
            cited_ids.append(doc_id)

    if not cited_ids:
        # Fallback: model didn't emit any valid markers
        return list(retrieved_docs)

    # Map IDs to documents, preserving citation order
    doc_by_id: dict[int, LexArticle] = {doc.id: doc for doc in retrieved_docs}
    return [doc_by_id[id_] for id_ in cited_ids if id_ in doc_by_id]
