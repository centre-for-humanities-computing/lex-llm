from ..api.orchestrator import Orchestrator
from ..api.event_models import WorkflowRunRequest
from ..api.connectors.openrouter_provider import OpenRouterProvider
from ..tools import search_knowledge_base, generate_response_with_sources
from ..prompts import ALPHA_V1_SYSTEM_PROMPT, ALPHA_V1_DEFERRAL_MESSAGE

def get_workflow(request: WorkflowRunRequest) -> Orchestrator:
    """Configures and returns the RAG workflow orchestrator using Gemma via OpenRouter."""

    return Orchestrator(
        request=request,
        steps=[
            search_knowledge_base(
                index_name="e5_small",
                top_k=10,
            ),
            generate_response_with_sources(
                llm_provider=OpenRouterProvider(
                    model="google/gemma-3-27b-it",
                    providers=["nebius/fp8"],  # Only use the nebius/fp8 provider
                ),
                system_prompt=ALPHA_V1_SYSTEM_PROMPT,
                deferral_message=ALPHA_V1_DEFERRAL_MESSAGE,
            ),
        ],
        context={"conversation_history": request.conversation_history},
    )


def get_metadata() -> dict:
    return {
        "workflow_id": "beta_workflow_v1",
        "name": "Beta Workflow v1",
        "description": "Version 1 of the beta workflow using Google Gemma 3 27B via OpenRouter and the e5 embedding model hosted locally. Performs a simple retrieval-augmented generation (RAG) using a knowledge base and outputs a source list.",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_input": {"type": "string"},
                "conversation_id": {"type": "string"},
                "conversation_history": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["user_input", "conversation_id", "conversation_history"],
        },
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
        "author": "Simon Enni",
        "version": "1.0.0",
        "tags": [
            "rag",
            "retrieval",
            "generation",
            "knowledge base",
            "openrouter",
            "gemma",
        ],
    }
