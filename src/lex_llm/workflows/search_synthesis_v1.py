"""Search & Synthesis workflow v1.

A search-and-synthesis interaction model that restructures answers into
5 sections: lead paragraph, body, definitions, interpretation, and sources.

Steps:
1. interpret_and_route — Combine query interpretation + scope routing
2. generate_deferral — Generate deferral message if out of scope
3. search_and_validate — Three-stage progressive hybrid retrieval with corrective-RAG
4. generate_answer_body — Generate the main answer body
5. generate_lead_paragraph — Generate the lead paragraph
6. ParallelStep([generate_definitions, generate_source_list]) — Definitions + sources in parallel

All LLM calls use OpenRouter with google/gemma-4-26B-A4B-it.
"""

from lex_llm.api.connectors.scaleway_provider import ScalewayProvider

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

# Shared LLM provider for all steps
_llm = ScalewayProvider(model="gemma-4-26b-a4b-it")


def get_workflow(request: WorkflowRunRequest) -> Orchestrator:
    """Configures and returns the Search & Synthesis v1 workflow orchestrator."""

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
        "workflow_id": "search_synthesis_v1",
        "name": "Search & Synthesis v1",
        "description": (
            "A search-and-synthesis workflow that restructures answers into "
            "5 sections: lead paragraph, body, definitions, interpretation, "
            "and sources. Uses combined query interpretation + routing, and a "
            "three-stage progressive retrieval strategy: simple hybrid search, "
            "intermediate keyword expansion, and advanced HyDE-based corrective-RAG. "
            "Parallel generation of definitions + source list. "
            "All LLM calls use Google Gemma 4 26B A4B via OpenRouter."
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
                "name": "Retrieval Cascade",
                "description": (
                    "Three-stage progressive hybrid retrieval: "
                    "(1) simple_retrieval with raw query, "
                    "(2) intermediate_retrieval with expanded keywords, "
                    "(3) advanced_retrieval with HyDE + corrective keyword broadening."
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
        "version": "1.0.0",
    }
