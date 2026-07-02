"""Chat Workflow v3 Mistral cloud version.

Latency-optimized variant of chat_v2_mistral that replaces the explicit
source-attribution LLM call with inline [^ID] citation markers. The markers
are extracted via regex post-processing — reducing total LLM calls from 3
to 2.
"""

from lex_llm.api.connectors.cortecs_provider import CortecsProvider
from datetime import datetime

from lex_llm.tools import interpret_and_route
from lex_llm.tools.generate_deferral import generate_deferral
from lex_llm.tools.hybrid_search import hybrid_search

from ..api.orchestrator import Orchestrator
from ..api.event_models import WorkflowRunRequest
from ..tools import generate_response_with_sources_v3
from ..prompts import get_deferral_message, get_system_prompt
from ..prompts_search_synthesis import _format_date as _format_date


_llm = CortecsProvider(
    model="mistral-small-2603", preference="speed", reasoning_effort="none"
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
            generate_response_with_sources_v3(
                llm_provider=_llm,
                system_prompt=get_system_prompt(
                    version="v3",
                    workflow_description=get_metadata()["description"],
                ),
                deferral_message=get_deferral_message(version="v3"),
                current_date=_format_date(datetime.today()),
            ),
        ],
        context={"conversation_history": request.conversation_history},
        use_clean_history=True,
    )


def get_metadata() -> dict:
    return {
        "workflow_id": "chat_v3_mistral",
        "name": "Chat v3 Mistral cloud version",
        "status": "active",
        "description": (
            "Latency-optimized variant of chat_v2_mistral using inline [^ID] "
            "citation markers instead of a separate source-attribution LLM call. "
            "Total LLM calls: 2 (routing + generation). Sources are extracted "
            "via regex post-processing. Uses Mistral models through Cortecs."
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
                "name": "Response Generation with inline citations",
                "description": "Formats a prompt using retrieved documents, streams "
                "the LLM response with inline [^ID] citation markers, then extracts "
                "cited sources via regex post-processing (no second LLM call).",
                "inputs": ["retrieved_docs", "conversation_history", "user_input"],
                "outputs": ["final_response", "used_sources"],
            },
        ],
        "author": "Simon Enni",
        "version": "3.0.0",
    }
