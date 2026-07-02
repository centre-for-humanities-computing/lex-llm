"""Editorial workflow v3 Mistral cloud version.

Latency-optimized variant of editorial_workflow_v2_mistral that replaces
the explicit source-attribution LLM call with inline [^ID] citation markers.
Total LLM calls reduced from 3 to 2.

Steps:
1. interpret_and_route — Combine query interpretation + scope routing
2. generate_deferral — Generate deferral message if out of scope
3. hybrid_search — Single-stage hybrid search
4. generate_lead_and_body_v3 — Single streaming call producing bold lead + body
   with [^ID] citations stripped from the client-facing stream
5. generate_source_list_v3 — Source attribution via regex extraction of [^ID] markers
"""

from lex_llm.api.connectors.cortecs_provider import CortecsProvider
from lex_llm.tools import hybrid_search

from ..api.orchestrator import Orchestrator
from ..api.event_models import WorkflowRunRequest
from ..tools import (
    interpret_and_route,
    generate_deferral,
    generate_lead_and_body_v3,
    generate_source_list_v3,
)
from ..prompts_search_synthesis import (
    get_lead_and_body_prompt_v3,
    _format_date as _format_date,
)
from datetime import date

_llm_small = CortecsProvider(
    model="mistral-nemo-instruct-2407", preference="speed", reasoning_effort="none"
)

_llm_large = CortecsProvider(
    model="mistral-small-2603", preference="speed", reasoning_effort="none"
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
            generate_lead_and_body_v3(
                llm_provider=_llm_large,
                system_prompt=get_lead_and_body_prompt_v3(
                    workflow_description=get_metadata().get("description"),
                ),
                current_date=_format_date(date.today()),
            ),
            generate_source_list_v3(),
        ],
        context={"conversation_history": request.conversation_history},
        use_clean_history=True,
    )


def get_metadata() -> dict:
    return {
        "workflow_id": "editorial_workflow_v3_mistral",
        "name": "Editorial workflow v3 Mistral cloud version",
        "status": "active",
        "description": (
            "A latency-optimized search-and-synthesis workflow that restructures "
            "answers into 4 sections: interpretation, lead paragraph, "
            "body, and sources. Uses a single-stage hybrid search, a single "
            "streaming LLM call for lead+body generation with inline [^ID] "
            "citations, and regex-based source extraction (total 2 LLM calls). "
            "Routing uses Mistral models."
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
                    "interpretation."
                ),
                "inputs": [
                    "user_input",
                    "query_interpretation",
                    "keywords",
                    "subqueries",
                ],
                "outputs": ["retrieved_docs", "retrieved_chunks"],
            },
            {
                "name": "Generate Lead & Body (v3 / stripped citations)",
                "description": (
                    "Single streaming LLM call that produces a bold Markdown lead "
                    "paragraph followed by an elaborating answer body with [^ID] "
                    "citation markers stripped during streaming for a clean "
                    "client-facing output."
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
                "name": "Generate Source List (v3 / regex)",
                "description": "Extracts cited sources from [^ID] markers in the "
                "answer body via regex — no LLM call needed.",
                "inputs": ["answer_body", "retrieved_docs"],
                "outputs": ["used_sources", "final_response"],
            },
        ],
        "author": "Simon Enni",
        "version": "3.0.0",
    }
