"""Simple search workflow v1.

The lightest-weight search endpoint: the raw user query is used
directly for both semantic and full-text search, fused with RRF.
No LLM calls — only the LexDB search API is contacted.

Steps:
1. simple_search — Hybrid search with the raw user query, no expansion
"""

from ..api.orchestrator import Orchestrator
from ..api.event_models import WorkflowRunRequest
from ..tools import hybrid_search


def get_workflow(request: WorkflowRunRequest) -> Orchestrator:
    """Configures and returns the Simple Search v1 workflow orchestrator."""

    return Orchestrator(
        request=request,
        steps=[
            hybrid_search(
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
        "workflow_id": "simple_search_v1",
        "name": "Simple Search v1",
        "status": "active",
        "description": (
            "The simplest search workflow: the raw user query is used "
            "directly for both semantic (vector) and full-text search, "
            "fused with Reciprocal Rank Fusion. No LLM calls and no query "
            "expansion — only the LexDB search API is contacted. Results are "
            "deduplicated at the article level with the most relevant chunk "
            "as a highlight."
        ),
        "steps": [
            {
                "name": "Simple Search",
                "description": (
                    "Hybrid search with the raw user query: batch vector "
                    "search + batch full-text search, fused with Reciprocal "
                    "Rank Fusion. No query expansion."
                ),
                "inputs": ["user_input"],
                "outputs": ["retrieved_docs", "search_results"],
            },
        ],
        "author": "Simon Enni",
        "version": "1.0.0",
    }
