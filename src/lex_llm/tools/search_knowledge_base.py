"""Knowledge base search tools for workflows."""

from typing import AsyncGenerator, Dict, Any, Callable
from ..api.event_emitter import EventEmitter
from ..api.connectors.lex_db_connector import LexDBConnector


def search_knowledge_base(
    index_name: str = "openai_large_3_sections",
    top_k: int = 10,
    search_method: str = "vector_search",
) -> Callable[[Dict[str, Any], EventEmitter], AsyncGenerator[None, None]]:
    """
    Creates a knowledge base search step with the specified parameters.

    Args:
        index_name: The name of the vector index to search
        top_k: Number of top results to retrieve
        search_method: Search method to use - one of: "vector_search", "hybrid_search", "hyde_search", "hybrid_hyde_search"
    Returns:
        An async generator function compatible with the Orchestrator
    """

    async def search_knowledge_base(
        context: Dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[None, None]:
        """Queries the KB and prepares sources for emission."""
        lex_db_connector = LexDBConnector()
        user_input = context.get("user_input", "")

        if search_method == "hybrid_search":
            documents = await lex_db_connector.hybrid_search(
                query=user_input, top_k=top_k, index_name=index_name
            )
        elif search_method == "hyde_search":
            documents = await lex_db_connector.hyde_search(
                query=user_input, top_k=top_k, index_name=index_name
            )
        elif search_method == "hybrid_hyde_search":
            documents = await lex_db_connector.hybrid_hyde_search(
                query=user_input, top_k=top_k, index_name=index_name
            )
        else:  # Default to vector_search
            documents = await lex_db_connector.vector_search(
                query=user_input, top_k=top_k, index_name=index_name
            )
        context["retrieved_docs"] = documents
        yield

    return search_knowledge_base
