from lex_llm.api.connectors.scaleway_provider import ScalewayProvider
from datetime import datetime

from ..api.orchestrator import Orchestrator
from ..api.event_models import WorkflowRunRequest
from ..tools import search_knowledge_base, generate_response_with_sources
from ..prompts import (
    get_deferral_message,
    get_system_prompt,
)


def get_workflow(request: WorkflowRunRequest) -> Orchestrator:
    """Configures and returns the RAG workflow orchestrator using Gemma via OpenRouter."""

    return Orchestrator(
        request=request,
        steps=[
            search_knowledge_base(
                index_name="article_embeddings_e5",
                top_k=10,
                search_method="hyde_search",
            ),
            generate_response_with_sources(
                llm_provider=ScalewayProvider(model="gemma-3-27b-it"),
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
        "workflow_id": "beta_workflow_v2_hyde",
        "name": "Beta Workflow v2 with different search methods",
        "description": (
            "Version 2 of the beta workflow using HyDE search method "
            "(generated a hypothetical documents and then performs semantic search)."
            "Uses Google Gemma 3 27B via OpenRouter and the multilingual e5 large embedding model hosted locally. "
        ),
        "steps": [
            {
                "name": "Knowledge Base Search",
                "description": "Queries the internal knowledge base for relevant documents using vector search.",
                "inputs": ["user_input"],
                "outputs": ["retrieved_docs", "sources"],
            },
            {
                "name": "Response Generation",
                "description": "Formats a prompt using the retrieved documents and streams the LLM response using Gemma.",
                "inputs": ["retrieved_docs", "conversation_history", "user_input"],
                "outputs": ["final_response"],
            },
        ],
        "author": "Zafar Hussain",
        "version": "2.0.0",
    }
