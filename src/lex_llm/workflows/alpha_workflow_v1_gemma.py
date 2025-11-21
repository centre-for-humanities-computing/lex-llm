from ..api.orchestrator import Orchestrator
from ..api.event_models import WorkflowRunRequest
from ..api.connectors.openrouter_provider import OpenRouterProvider
from ..tools import create_kb_search_step, create_response_generation_step
from ..prompts import ALPHA_V1_SYSTEM_PROMPT, ALPHA_V1_DEFERRAL_MESSAGE


# This function is the entry point called by the API route
def get_workflow(request: WorkflowRunRequest) -> Orchestrator:
    """Configures and returns the RAG workflow orchestrator using Gemma via OpenRouter."""

    return Orchestrator(
        request=request,
        steps=[
            create_kb_search_step(
                index_name="openai_large_3_sections",
                top_k=10,
            ),
            create_response_generation_step(
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
        "workflow_id": "alpha_workflow_v1_gemma",
        "name": "Alpha Workflow v1 (Gemma)",
        "description": "Version 1 of the workflow using Google Gemma 3 27B via OpenRouter. Performs a simple retrieval-augmented generation (RAG) using a knowledge base and outputs a source list.",
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
