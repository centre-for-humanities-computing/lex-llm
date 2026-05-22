"""Search-only workflow v1.

A lightweight search endpoint workflow that returns a list of matching
articles without summarization or answer generation.

Steps:
1. search_with_expansion — Query expansion + hybrid search with RRF fusion

All LLM calls use Cortecs.ai with google/gemma-4-26B-A4B-it.
"""

from lex_llm.api.connectors.scaleway_provider import ScalewayProvider

from ..api.orchestrator import Orchestrator
from ..api.event_models import WorkflowRunRequest
from ..tools.search_with_expansion import search_with_expansion

# Shared LLM provider for all steps
_llm = ScalewayProvider(model="gemma-4-26b-a4b-it")


def get_workflow(request: WorkflowRunRequest) -> Orchestrator:
    """Configures and returns the Intermediate Search v1 workflow orchestrator."""

    return Orchestrator(
        request=request,
        steps=[
            search_with_expansion(
                llm_provider=_llm,
                index_name="article_embeddings_e5",
                top_k=100,
                top_k_semantic=100,
                top_k_fts=100,
                rrf_k=60,
            ),
        ],
        context={"conversation_history": request.conversation_history},
    )


def get_metadata() -> dict:
    return {
        "workflow_id": "intermediate_search_v1",
        "name": "Intermediate Search v1",
        "description": (
            "A search-only workflow that returns a list of matching articles "
            "without summarization or answer generation. Uses query expansion "
            "(semantic subqueries + keyword queries) and hybrid retrieval with "
            "Reciprocal Rank Fusion. Results are deduplicated at the article "
            "level with the most relevant chunk as a highlight. "
            "All LLM calls use Google Gemma 4 26B A4B via Scaleway."
        ),
        "steps": [
            {
                "name": "Search with Expansion",
                "description": (
                    "Query expansion via LLM (semantic subqueries + keyword queries), "
                    "followed by hybrid search: batch vector search + batch full-text "
                    "search, fused with Reciprocal Rank Fusion."
                ),
                "inputs": ["user_input"],
                "outputs": ["retrieved_docs", "search_queries"],
            },
        ],
        "author": "Simon Enni",
        "version": "1.0.0",
    }
