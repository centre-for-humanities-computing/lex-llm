"""Beta Workflow v4 (fast).

Fast variant of beta_workflow_v3 with latency optimizations:
- Two-stage retrieval cascade (no HyDE / advanced expansion)
- Merged eval+expand LLM call between stages
- Cumulative RRF-fused chunk pool across both stages
"""

from lex_llm.api.connectors.scaleway_provider import ScalewayProvider
from datetime import datetime

from lex_llm.tools import interpret_and_route
from lex_llm.tools.generate_deferral import generate_deferral
from lex_llm.tools.retrieval_cascade_fast import retrieval_cascade_fast

from ..api.orchestrator import Orchestrator
from ..api.event_models import WorkflowRunRequest
from ..tools import generate_response_with_sources
from ..prompts import get_deferral_message, get_system_prompt


# Shared LLM provider for all steps
_llm = ScalewayProvider(model="gemma-4-26b-a4b-it")


def get_workflow(request: WorkflowRunRequest) -> Orchestrator:
    """Configures and returns the fast RAG workflow orchestrator using Gemma via Scaleway."""

    return Orchestrator(
        request=request,
        steps=[
            interpret_and_route(llm_provider=_llm),
            generate_deferral(llm_provider=_llm),
            retrieval_cascade_fast(
                llm_provider=_llm,
                index_name="article_embeddings_e5",
                top_k=25,
                top_k_semantic=40,
                top_k_fts=40,
                rrf_k=60,
            ),
            generate_response_with_sources(
                llm_provider=_llm,
                system_prompt=get_system_prompt(
                    version="alpha_v1",
                    current_date=datetime.today(),
                    workflow_description=get_metadata()["description"],
                ),
                deferral_message=get_deferral_message(version="alpha_v1"),
            ),
        ],
        context={"conversation_history": request.conversation_history},
    )


def get_metadata() -> dict:
    return {
        "workflow_id": "beta_workflow_v4",
        "name": "Beta Workflow v4 (fast)",
        "status": "deprecated",
        "description": (
            "Fast variant of Beta Workflow v3 using Google Gemma 4 26B "
            "via Scaleway and the multilingual e5 large embedding model. "
            "Two-stage retrieval cascade (simple → corrective intermediate) "
            "with merged eval+expand LLM call between stages and cumulative "
            "RRF-fused chunk pool. No HyDE / advanced expansion."
        ),
        "steps": [
            {
                "name": "Interpret & Route",
                "description": "Interprets the user query and determines if it's within scope. Returns lists of keywords and subqueries for retrieval.",
                "inputs": ["user_input"],
                "outputs": [
                    "query_interpretation",
                    "is_in_scope",
                    "routing_reason",
                    "keywords",
                    "subqueries",
                ],
            },
            {
                "name": "Generate Deferral",
                "description": "Generates a deferral message if the query is out of scope.",
                "inputs": ["is_in_scope", "routing_reason"],
                "outputs": ["final_response"],
            },
            {
                "name": "Retrieval Cascade (fast)",
                "description": (
                    "Two-stage progressive hybrid retrieval: "
                    "(1) simple_retrieval with keywords/subqueries from interpretation, "
                    "(2) corrective intermediate search driven by a merged eval+expand LLM call. "
                    "Cumulative raw result lists are RRF-fused across both stages."
                ),
                "inputs": [
                    "user_input",
                    "query_interpretation",
                    "keywords",
                    "subqueries",
                ],
                "outputs": ["retrieved_docs", "insufficient_context"],
            },
            {
                "name": "Response Generation with sources",
                "description": "Formats a prompt using the retrieved documents and streams the LLM response using Gemma.",
                "inputs": ["retrieved_docs", "conversation_history", "user_input"],
                "outputs": ["final_response"],
            },
        ],
        "author": "Simon Enni",
        "version": "1.0.0",
    }
