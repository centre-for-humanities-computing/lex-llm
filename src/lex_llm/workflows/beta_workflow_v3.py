from lex_llm.api.connectors.scaleway_provider import ScalewayProvider
from datetime import datetime

from lex_llm.tools import interpret_and_route
from lex_llm.tools.generate_deferral import generate_deferral
from lex_llm.tools.retrieval_cascade import retrieval_cascade

from ..api.orchestrator import Orchestrator
from ..api.event_models import WorkflowRunRequest
from ..tools import generate_response_with_sources
from ..prompts import get_deferral_message, get_system_prompt


# Shared LLM provider for all steps
_llm = ScalewayProvider(model="gemma-4-26b-a4b-it")


def get_workflow(request: WorkflowRunRequest) -> Orchestrator:
    """Configures and returns the RAG workflow orchestrator using Gemma via OpenRouter."""

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
        "workflow_id": "beta_workflow_v3",
        "name": "Beta Workflow v3",
        "description": (
            "Version 3 of the beta workflow using Google Gemma 4 26B "
            "via Scaleway and the multilingual e5 large embedding model hosted locally. "
            "Performs a retrieval cascade and generates a chat response with sources."
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
                "name": "Response Generation with sources",
                "description": "Formats a prompt using the retrieved documents and streams the LLM response using Gemma.",
                "inputs": ["retrieved_docs", "conversation_history", "user_input"],
                "outputs": ["final_response"],
            },
        ],
        "author": "Simon Enni",
        "version": "1.0.0",
    }
