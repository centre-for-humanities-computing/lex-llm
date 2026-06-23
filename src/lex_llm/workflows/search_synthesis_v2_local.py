"""Search & Synthesis workflow v2 (fast) local version.

A latency-optimized variant of search_synthesis_v1 that:
- Uses a fast two-stage retrieval cascade (no HyDE, merged eval+expand)
- Merges lead paragraph + answer body into a single streaming LLM call
  (bold Markdown lead followed by elaborating body, streamed as text_chunk)
- Drops the definitions step entirely

Steps:
1. interpret_and_route — Combine query interpretation + scope routing
2. generate_deferral — Generate deferral message if out of scope
3. retrieval_cascade_fast — Two-stage progressive hybrid retrieval with cumulative RRF
4. generate_lead_and_body — Single streaming call producing bold lead + body
5. generate_source_list — Source attribution for conversation history
"""

import os

from lex_llm.api.connectors.dgx_provider import DGXProvider
from lex_llm.api.connectors.routing_llm_provider import RoutingLLMProvider
from lex_llm.api.connectors.scaleway_provider import ScalewayProvider
from lex_llm.api.connectors.vllm_load_probe import VLLMLoadProbe

from ..api.orchestrator import Orchestrator
from ..api.event_models import WorkflowRunRequest
from ..tools import (
    interpret_and_route,
    generate_deferral,
    retrieval_cascade_fast,
    generate_lead_and_body,
    generate_source_list,
)
from ..prompts_search_synthesis import get_lead_and_body_prompt
from datetime import date

# LLM provider for large model
_model_name_large = "gemma-4-26B-A4B-it"
_metrics_url_large = f"{os.environ['METRICS_SERVER_URL']}/metrics/{_model_name_large}"
_probe_large = VLLMLoadProbe(_metrics_url_large, model_name=_model_name_large)

_llm_large = RoutingLLMProvider(
    primary=DGXProvider(model=_model_name_large),
    fallback=ScalewayProvider(model="gemma-4-26b-a4b-it"),
    probe=_probe_large,
)

# LLM provider for small model (used for routing/interpretation)
_model_name_small = "gemma-4-E2B-it"
_metrics_url_small = f"{os.environ['METRICS_SERVER_URL']}/metrics/{_model_name_small}"
_probe_small = VLLMLoadProbe(_metrics_url_small, model_name=_model_name_small)

_llm_small = RoutingLLMProvider(
    primary=DGXProvider(model=_model_name_small),
    fallback=ScalewayProvider(model="gemma-4-26b-a4b-it"),
    probe=_probe_small,
)


def get_workflow(request: WorkflowRunRequest) -> Orchestrator:
    """Configures and returns the Search & Synthesis v2 (fast) workflow orchestrator."""

    return Orchestrator(
        request=request,
        steps=[
            interpret_and_route(llm_provider=_llm_small),
            generate_deferral(llm_provider=_llm_large),
            retrieval_cascade_fast(
                llm_provider=_llm_large,
                index_name="article_embeddings_e5",
                top_k=25,
                top_k_semantic=40,
                top_k_fts=40,
                rrf_k=60,
            ),
            generate_lead_and_body(
                llm_provider=_llm_large,
                system_prompt=get_lead_and_body_prompt(
                    date.today(),
                    workflow_description=get_metadata().get("description"),
                ),
            ),
            generate_source_list(llm_provider=_llm_large),
        ],
        context={"conversation_history": request.conversation_history},
    )


def get_metadata() -> dict:
    return {
        "workflow_id": "search_synthesis_v2_local",
        "name": "Search & Synthesis v2 (fast) local version",
        "status": "deprecated",
        "description": (
            "A latency-optimized search-and-synthesis workflow that restructures "
            "answers into 4 sections: interpretation, lead paragraph (bold), "
            "body, and sources. Uses a two-stage fast retrieval cascade with "
            "merged eval+expand and cumulative RRF, a single streaming LLM call "
            "for lead+body generation, and drops the definitions step. "
            "Routing LLM calls use Google Gemma 4 E2B and other steps use Google "
            "Gemma 4 26B A4B via local DGX Spark - both fall back to 26B A4B on Scaleway for backup."
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
                "name": "Generate Lead & Body (merged)",
                "description": (
                    "Single streaming LLM call that produces a bold Markdown lead "
                    "paragraph followed by an elaborating answer body. Streams as "
                    "text_chunk events."
                ),
                "inputs": [
                    "retrieved_docs",
                    "retrieved_chunks",
                    "conversation_history",
                    "user_input",
                    "query_interpretation",
                ],
                "outputs": ["final_response", "answer_body", "lead_paragraph"],
            },
            {
                "name": "Generate Source List",
                "description": "Attributes which sources were used in the answer.",
                "inputs": ["answer_body", "retrieved_docs"],
                "outputs": ["used_sources", "system_prompt", "final_response"],
            },
        ],
        "author": "Simon Enni",
        "version": "1.0.0",
    }
