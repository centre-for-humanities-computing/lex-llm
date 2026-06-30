"""Editorial workflow v2 local version.

A further latency-optimized variant of search_synthesis_v2 that:
- Uses a single stage hybrid search with no separate corrective retrieval stage, relying on the LLM to effectively leverage keywords + subqueries from interpretation in one pass

Steps:
1. interpret_and_route — Combine query interpretation + scope routing
2. generate_deferral — Generate deferral message if out of scope
3. hybrid_search — Single-stage hybrid search with no separate corrective retrieval stage
4. generate_lead_and_body — Single streaming call producing bold lead + body
5. generate_source_list — Source attribution for conversation history
"""

import os

from lex_llm.api.connectors.cortecs_provider import CortecsProvider
from lex_llm.api.connectors.dgx_provider import DGXProvider
from lex_llm.api.connectors.routing_llm_provider import RoutingLLMProvider
from lex_llm.api.connectors.vllm_load_probe import VLLMLoadProbe
from lex_llm.tools import hybrid_search

from ..api.orchestrator import Orchestrator
from ..api.event_models import WorkflowRunRequest
from ..tools import (
    interpret_and_route,
    generate_deferral,
    generate_lead_and_body_v2,
    generate_source_list_v2,
)
from ..prompts_search_synthesis import (
    get_lead_and_body_prompt_v2,
    _format_date as _format_date,
)
from datetime import date

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
            generate_lead_and_body_v2(
                llm_provider=_llm_large,
                system_prompt=get_lead_and_body_prompt_v2(
                    workflow_description=get_metadata().get("description"),
                ),
                current_date=_format_date(date.today()),
            ),
            generate_source_list_v2(llm_provider=_llm_large),
        ],
        context={"conversation_history": request.conversation_history},
        use_clean_history=True,
    )


def get_metadata() -> dict:
    return {
        "workflow_id": "editorial_workflow_v2_local",
        "name": "Editorial workflow v2 local version",
        "status": "active",
        "description": (
            "A latency-optimized search-and-synthesis workflow that restructures "
            "answers into 4 sections: interpretation, lead paragraph, "
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
        "version": "2.0.0",
    }
