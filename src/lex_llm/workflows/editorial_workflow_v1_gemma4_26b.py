"""Editorial workflow v1 Gemma 4 26B cloud version.

A further latency-optimized variant of search_synthesis_v2 that:
- Uses a single stage hybrid search with no separate corrective retrieval stage, relying on the LLM to effectively leverage keywords + subqueries from interpretation in one pass

Steps:
1. interpret_and_route — Combine query interpretation + scope routing
2. generate_deferral — Generate deferral message if out of scope
3. hybrid_search — Single-stage hybrid search with no separate corrective retrieval stage
4. generate_lead_and_body — Single streaming call producing bold lead + body
5. generate_source_list — Source attribution for conversation history
"""

from lex_llm.api.connectors.cortecs_provider import CortecsProvider
from lex_llm.tools import hybrid_search

from ..api.orchestrator import Orchestrator
from ..api.event_models import WorkflowRunRequest
from ..tools import (
    interpret_and_route,
    generate_deferral,
    generate_lead_and_body,
    generate_source_list,
)
from ..prompts_search_synthesis import get_lead_and_body_prompt
from datetime import date

_llm = CortecsProvider(
    model="gemma-4-26b-a4b-it", preference="speed", reasoning_effort="none"
)


def get_workflow(request: WorkflowRunRequest) -> Orchestrator:
    return Orchestrator(
        request=request,
        steps=[
            interpret_and_route(llm_provider=_llm),
            generate_deferral(llm_provider=_llm),
            hybrid_search(
                index_name="article_embeddings_e5",
                top_k=20,
                top_k_semantic=30,
                top_k_fts=30,
                rrf_k=60,
            ),
            generate_lead_and_body(
                llm_provider=_llm,
                system_prompt=get_lead_and_body_prompt(
                    date.today(),
                    workflow_description=get_metadata().get("description"),
                ),
            ),
            generate_source_list(llm_provider=_llm),
        ],
        context={"conversation_history": request.conversation_history},
    )


def get_metadata() -> dict:
    return {
        "workflow_id": "editorial_workflow_v1_gemma4_26b",
        "name": "Editorial workflow v1 Gemma 4 26B cloud version",
        "status": "active",
        "description": (
            "A latency-optimized search-and-synthesis workflow that restructures "
            "answers into 4 sections: interpretation, lead paragraph, "
            "body, and sources. Uses a two-stage fast retrieval cascade with "
            "merged eval+expand and cumulative RRF, a single streaming LLM call "
            "for lead+body generation, and drops the definitions step. "
            "Uses Google Gemma 4 26B A4B for all LLM calls."
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
