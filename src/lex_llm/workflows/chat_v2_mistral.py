"""Chat Workflow v2 Mistral cloud version.

Faster variant of beta_workflow_v4 with latency optimizations:
- Single stage routing+retrieval.
- Deferral roped into routing+retrieval.
"""

from lex_llm.api.connectors.cortecs_provider import CortecsProvider
from datetime import datetime

from lex_llm.tools import interpret_and_route
from lex_llm.tools.generate_deferral import generate_deferral
from lex_llm.tools.hybrid_search import hybrid_search

from ..api.orchestrator import Orchestrator
from ..api.event_models import WorkflowRunRequest
from ..tools import generate_response_with_sources_v2
from ..prompts import get_deferral_message, get_system_prompt
from ..prompts_search_synthesis import _format_date as _format_date

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
            generate_response_with_sources_v2(
                llm_provider=_llm_large,
                system_prompt=get_system_prompt(
                    version="v2",
                    workflow_description=get_metadata()["description"],
                ),
                deferral_message=get_deferral_message(version="v2"),
                current_date=_format_date(datetime.today()),
            ),
        ],
        context={"conversation_history": request.conversation_history},
        use_clean_history=True,
    )


def get_metadata() -> dict:
    return {
        "workflow_id": "chat_v2_mistral",
        "name": "Chat v2 Mistral cloud version",
        "status": "active",
        "description": (
            "Faster variant of Beta Workflow v4 using Mistral models through Cortecs. "
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
                "description": "Formats a prompt using the retrieved documents and streams the LLM response using Mistral models.",
                "inputs": ["retrieved_docs", "conversation_history", "user_input"],
                "outputs": ["final_response"],
            },
        ],
        "author": "Simon Enni",
        "version": "2.0.0",
    }
