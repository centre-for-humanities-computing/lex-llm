"""Chat Workflow v1 local version.

Faster variant of beta_workflow_v4 with latency optimizations:
- Single stage routing+retrieval.
- Deferral roped into routing+retrieval.
"""

import os

from lex_llm.api.connectors.cortecs_provider import CortecsProvider
from lex_llm.api.connectors.dgx_provider import DGXProvider
from lex_llm.api.connectors.routing_llm_provider import RoutingLLMProvider
from datetime import datetime

from lex_llm.api.connectors.vllm_load_probe import VLLMLoadProbe
from lex_llm.tools import interpret_and_route
from lex_llm.tools.generate_deferral import generate_deferral
from lex_llm.tools.hybrid_search import hybrid_search

from ..api.orchestrator import Orchestrator
from ..api.event_models import WorkflowRunRequest
from ..tools import generate_response_with_sources
from ..prompts import get_deferral_message, get_system_prompt


# LLM provider for large model
_model_name_large = "gemma-4-26B-A4B-it"
_metrics_url_large = f"{os.environ['METRICS_SERVER_URL']}/metrics/{_model_name_large}"
_probe_large = VLLMLoadProbe(_metrics_url_large, model_name=_model_name_large)

_llm_large = RoutingLLMProvider(
    primary=DGXProvider(model=_model_name_large),
    fallback=CortecsProvider(
        model="gemma-4-26b-a4b-it", preference="speed", reasoning_effort="none"
    ),
    probe=_probe_large,
)

# LLM provider for small model (used for routing/interpretation)
_model_name_small = "gemma-4-E2B-it"
_metrics_url_small = f"{os.environ['METRICS_SERVER_URL']}/metrics/{_model_name_small}"
_probe_small = VLLMLoadProbe(_metrics_url_small, model_name=_model_name_small)

_llm_small = RoutingLLMProvider(
    primary=DGXProvider(model=_model_name_small),
    fallback=CortecsProvider(
        model="gemma-4-26b-a4b-it", preference="speed", reasoning_effort="none"
    ),
    probe=_probe_small,
)


def get_workflow(request: WorkflowRunRequest) -> Orchestrator:
    return Orchestrator(
        request=request,
        steps=[
            interpret_and_route(llm_provider=_llm_small),
            generate_deferral(llm_provider=_llm_small),
            hybrid_search(
                index_name="article_embeddings_e5",
                top_k=20,
                top_k_semantic=30,
                top_k_fts=30,
                rrf_k=60,
            ),
            generate_response_with_sources(
                llm_provider=_llm_large,
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
        "workflow_id": "chat_v1_local",
        "name": "Chat v1 local version",
        "status": "active",
        "description": (
            "Faster variant of Beta Workflow v4 using Google Gemma 4 26B and 4 "
            "E2B via local DGX Spark with Gemma 4 26B A4B on Scaleway as backup. "
            "Generates a response with sources in a single streaming LLM call, "
            "and uses the multilingual e5 large embedding model for search. "
            "Single-stage hybrid search using keywords and subqueries from "
            "interpretation, with no separate corrective retrieval stage. "
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
                "name": "Hybrid Search",
                "description": (
                    "Single-stage hybrid search using keywords and subqueries from "
                    "interpretation, with no separate corrective retrieval stage. "
                ),
                "inputs": [
                    "user_input",
                    "query_interpretation",
                    "keywords",
                    "subqueries",
                ],
                "outputs": ["retrieved_docs"],
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
