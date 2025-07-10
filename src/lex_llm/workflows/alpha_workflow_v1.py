from typing import AsyncGenerator, Dict, Any
from ..api.orchestrator import Orchestrator
from ..api.event_emitter import EventEmitter
from ..api.event_models import WorkflowRunRequest, Source
from ..api.connectors.lex_db_connector import LexDBConnector
from ..api.connectors.openai_provider import OpenAIProvider


# Step 1: Search the knowledge base
async def search_knowledge_base(
    context: Dict[str, Any], emitter: EventEmitter
) -> AsyncGenerator[str, None]:
    """Queries the KB and prepares sources for emission."""
    lex_db_connector = LexDBConnector()
    user_input = context.get("user_input", "")

    documents = await lex_db_connector.vector_search(query=user_input, top_k=10)
    context["retrieved_docs"] = documents

    sources = [Source(id=doc.id, title=doc.title, url=doc.url) for doc in documents]
    # Emits the sources list immediately after retrieval
    yield emitter.sources(sources)


# Step 2: Generate the response using the retrieved documents
async def generate_response(
    context: Dict[str, Any], emitter: EventEmitter
) -> AsyncGenerator[str, None]:
    """Formats the prompt and streams the LLM response."""
    llm_provider = OpenAIProvider()

    # Simple prompt engineering
    docs_text = "\n\n".join(
        [
            "Title: " + doc.title + "\nContent: " + doc.text
            for doc in context.get("retrieved_docs", [])
        ]
    )

    deferral_message = "Jeg beklager, men jeg er ikke i stand til at besvare dit spørgsmål ud fra mine kilder."
    history_as_dicts = [
        msg.model_dump() if hasattr(msg, "model_dump") else msg
        for msg in context.get("conversation_history", [])
    ]
    if not history_as_dicts:
        # We're starting a new conversation
        system_prompt = f"""You are an assistant, helping a user browse the Danish Lexicon. Read the search results carefully and answer the user's question in Danish. 
        It is imperative that you ONLY make factual statements that are directly supported by the sources. 
        In the event that the conversation continues over multiple messages, you are allowed to use and reference sources from previous answers. 
        If you reference a source in the text make sure to do so with a correctly formatted markdown link.
        Example: [Frederik 6., 1768-1839](https://danmarkshistorien.lex.dk/Frederik_6.,_1768-1839)
        Only refer to sources in the text if they explicitly and transparently back up the statement made in the text.
        Do not attach a list of sources to the end of the response - this will be done programmatically.
        If the sources do not support an answer to the users query use this answer: {deferral_message}
        Format your response as correct markdown.

        Search results:
        {docs_text}
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context.get("user_input", "")},
        ]
    else:
        system_prompt = f"New search results for user input: \n{docs_text}"
        messages = history_as_dicts + [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context.get("user_input", "")},
        ]

    full_response = ""
    async for chunk in llm_provider.generate_stream(messages):  # type: ignore
        full_response += chunk
        yield emitter.text_chunk(chunk)  # Stream chunks to the client

    context["final_response"] = full_response


# This function is the entry point called by the API route
def get_workflow(request: WorkflowRunRequest) -> Orchestrator:
    """Configures and returns the RAG workflow orchestrator."""
    return Orchestrator(
        request=request,
        steps=[
            search_knowledge_base,
            generate_response,
        ],
        context={"conversation_history": request.conversation_history},
    )


def get_metadata() -> dict:
    return {
        "workflow_id": "alpha_workflow_v1",
        "name": "Alpha Workflow v1",
        "description": "Version 1 of the workflow for the alpha version. Performs a simple retrieval-augmented generation (RAG) using a knowledge base and an LLM and outputs a source list.",
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
                "description": "Formats a prompt using the retrieved documents and streams the LLM response.",
                "inputs": ["retrieved_docs", "conversation_history", "user_input"],
                "outputs": ["final_response"],
            },
        ],
        "author": "Simon Enni",
        "version": "1.0.0",
        "tags": ["rag", "retrieval", "generation", "knowledge base", "openai"],
    }
