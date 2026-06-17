"""Search & Synthesis workflow v2.

A search-and-synthesis interaction model that restructures answers into
5 sections: lead paragraph, body, definitions, interpretation, and sources.

Steps:
1. interpret_and_route — Combine query interpretation + scope routing
2. generate_deferral — Generate deferral message if out of scope
3. search_and_validate — Three-stage progressive hybrid retrieval with corrective-RAG
4. generate_answer_body — Generate the main answer body
5. generate_lead_paragraph — Generate the lead paragraph
6. ParallelStep([generate_definitions, generate_source_list]) — Definitions + sources in parallel

All LLM calls use the DGX Spark with gemma-4-26B-A4B-it and fallback to Scaleway.
"""

from lex_llm.api.connectors.dgx_provider import DGXProvider
from lex_llm.api.connectors.routing_llm_provider import RoutingLLMProvider
from lex_llm.api.connectors.scaleway_provider import ScalewayProvider
from lex_llm.api.connectors.vllm_load_probe import VLLMLoadProbe

from ..api.orchestrator import Orchestrator, ParallelStep
from ..api.event_models import WorkflowRunRequest
from ..tools import (
    interpret_and_route,
    generate_deferral,
    retrieval_cascade,
    generate_answer_body,
    generate_lead_paragraph,
    generate_definitions,
    generate_source_list,
)
from ..prompts_search_synthesis import get_answer_body_prompt
from datetime import date
import os

# Shared LLM provider for all steps
_model_name = "gemma-4-26B-A4B-it"
_metrics_url = f"{os.environ['METRICS_SERVER_URL']}/metrics/{_model_name}"
_probe = VLLMLoadProbe(_metrics_url, model_name=_model_name)

_llm = RoutingLLMProvider(
    primary=DGXProvider(model=_model_name),
    fallback=ScalewayProvider(model="gemma-4-26b-a4b-it"),
    probe=_probe,
)


def get_workflow(request: WorkflowRunRequest) -> Orchestrator:
    """Configures and returns the Search & Synthesis v2 workflow orchestrator."""

    return Orchestrator(
        request=request,
        steps=[
            interpret_and_route(llm_provider=_llm),
            generate_deferral(llm_provider=_llm),
            retrieval_cascade(
                llm_provider=_llm,
                index_name="article_embeddings_e5",
                top_k=25,
                top_k_semantic=40,
                top_k_fts=40,
                rrf_k=60,
            ),
            generate_answer_body(
                llm_provider=_llm,
                system_prompt=get_answer_body_prompt(
                    date.today(), workflow_description=get_metadata().get("description")
                ),
            ),  # system prompt can be empty or customized as needed
            generate_lead_paragraph(llm_provider=_llm),
            (
                ParallelStep(
                    steps=[
                        generate_definitions(llm_provider=_llm),
                        generate_source_list(llm_provider=_llm),
                    ],
                    label="definitions_and_sources",
                ),
                "Genererer definitioner og kildeliste parallelt",
            ),
        ],
        context={"conversation_history": request.conversation_history},
    )


def get_metadata() -> dict:
    return {
        "workflow_id": "search_synthesis_v2",
        "name": "Search & Synthesis v2",
        "description": (
            "A search-and-synthesis workflow that restructures answers into "
            "5 sections: lead paragraph, body, definitions, interpretation, "
            "and sources. Uses combined query interpretation + routing, and a "
            "three-stage progressive retrieval strategy: simple hybrid search, "
            "intermediate keyword expansion, and advanced HyDE-based corrective-RAG. "
            "Parallel generation of definitions + source list. "
            "All LLM calls use Google Gemma 4 26B A4B via the DGX Spark with a fallback to Scaleway."
        ),
        "steps": [
            {
                "name": "Interpret & Route",
                "description": "Interprets the user query and determines if it's within scope.",
                "inputs": ["user_input"],
                "outputs": ["query_interpretation", "is_in_scope", "routing_reason"],
            },
            {
                "name": "Generate Deferral",
                "description": "Generates a deferral message if the query is out of scope.",
                "inputs": ["is_in_scope", "routing_reason"],
                "outputs": ["final_response"],
            },
            {
                "name": "Retrieval Cascade",
                "description": (
                    "Three-stage progressive hybrid retrieval: "
                    "(1) simple_retrieval with raw query, "
                    "(2) intermediate_retrieval with expanded keywords, "
                    "(3) advanced_retrieval with HyDE + corrective keyword broadening."
                ),
                "inputs": ["user_input", "query_interpretation"],
                "outputs": ["retrieved_docs", "insufficient_context"],
            },
            {
                "name": "Generate Answer Body",
                "description": "Generates the main answer body from retrieved documents.",
                "inputs": ["retrieved_docs", "user_input", "query_interpretation"],
                "outputs": ["answer_body"],
            },
            {
                "name": "Generate Lead Paragraph",
                "description": "Generates a concise lead paragraph summarizing the answer.",
                "inputs": ["answer_body", "user_input"],
                "outputs": ["lead_paragraph"],
            },
            {
                "name": "Definitions & Sources (parallel)",
                "description": (
                    "Generates key term definitions and attributes sources in parallel."
                ),
                "inputs": ["answer_body", "retrieved_docs"],
                "outputs": ["definitions", "used_sources", "final_response"],
            },
        ],
        "author": "Simon Enni",
        "version": "2.0.0",
    }
