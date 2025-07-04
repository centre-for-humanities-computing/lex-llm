import argparse
import os
from smolagents import CodeAgent, LiteLLMModel, tool  # type: ignore
from lex_db_api.configuration import Configuration
from lex_db_api.models.vector_search_request import VectorSearchRequest
from lex_db_api.api_client import ApiClient
from lex_db_api.api.lex_db_api import LexDbApi
from lex_db_api.models.search_results import SearchResults
from lex_db_api.models.vector_search_results import VectorSearchResults


model = LiteLLMModel(
    model_id="openai/gpt-4.1",
    api_key=os.environ[
        "OPENAI_API_KEY"
    ],  # Switch to the API key for the server you're targeting.
)

lexdb_client = ApiClient(
    configuration=Configuration(host=os.getenv("DB_HOST", "http://localhost:8000"))
)
lexdb_api = LexDbApi(api_client=lexdb_client)


@tool
def full_text_search_tool(query: str) -> str:
    """
    Perform a full-text search in the LexDB.

    Args:
        query (str): The search query.

    Returns:
        str: The search results.
    """
    try:
        results = search_lex_db(query)
        if not results.entries:
            return "No results found."
        return "\n".join(
            [
                f"Source {i}: {result.xhtml_md}"
                for i, result in enumerate(results.entries)
            ]
        )
    except Exception as e:
        print(f"Error during LexDB full-text search: {e}")
        raise (e)


def search_lex_db(query: str) -> SearchResults:
    """
    Search the LexDB for a given query.

    Args:
        query (str): The search query.

    Returns:
        str: The search results.
    """
    try:
        results = lexdb_api.get_articles(query, limit=5)
        return results
    except Exception as e:
        print(f"Error during LexDB search: {e}")
        raise (e)


@tool
def vector_search_tool(query: str, top_k: int = 5) -> str:
    """
    Perform a vector search in the LexDB.

    Args:
        query (str): The search query.
        top_k (int): The number of top results to return.

    Returns:
        str: The vector search results.
    """
    try:
        results = vector_search_lex_db(query, top_k)
        if not results.results:
            return "No results found."
        return "\n".join(
            [
                f"Source {i}: {result.chunk_text}"
                for i, result in enumerate(results.results)
            ]
        )
    except Exception as e:
        print(f"Error during LexDB vector search: {e}")
        raise (e)


def vector_search_lex_db(query: str, top_k: int = 5) -> VectorSearchResults:
    """Perform a vector search in the LexDB.
    Args:
        query (str): The search query.
        top_k (int): The number of top results to return.
    Returns:
        VectorSearchResults: The vector search results.
    """
    try:
        req = VectorSearchRequest(
            query_text=query,
            top_k=top_k,
        )
        results = lexdb_api.vector_search("small_003", req)
        return results
    except Exception as e:
        print(f"Error during LexDB vector search: {e}")
        raise (e)


agent = CodeAgent(tools=[vector_search_tool], model=model, add_base_tools=False)


def main() -> None:
    """
    Main function to test the OpenAI server model with a simple prompt.
    """
    parser = argparse.ArgumentParser(
        description="Create an empty vector index structure (without populating it)"
    )

    parser.add_argument(
        "mode",
        type=str,
        choices=["rag", "agent"],
    )

    parser.add_argument(
        "prompt",
        type=str,
        help="The prompt to send to the OpenAI server model.",
    )
    args = parser.parse_args()

    if args.mode == "rag":
        vector_search_result = vector_search_lex_db(args.prompt)
        print("Vector Search Result:", vector_search_result)
        if not vector_search_result.results:
            print("No results found.")
            vector_search_result_str = "No results found."
        else:
            vector_search_result_str = "\n".join(
                [
                    f"Source {i}: {result.chunk_text}"
                    for i, result in enumerate(vector_search_result.results)
                ]
            )
        rag_prompt = [
            {
                "role": "system",
                "content": "You are an assistant, helping a user browse the Danish Lexicon. Read the search results carefully and answer the user's question in Danish.",
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"User prompt: {args.prompt}"}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Search results for '{args.prompt}': {vector_search_result_str}\n\n",
                    }
                ],
            },
        ]
        result = model.generate(
            messages=rag_prompt,
            max_tokens=1000,
            temperature=0.0,
        )
        print("\n\nRAG Result:", result)
    if args.mode == "agent":
        agent_prompt: str = "You are an agent that can search the Danish Lexicon. You can use the search_lex_db tool to search for information."
        agent_prompt += f"User prompt: {args.prompt}"
        try:
            agent.run(
                agent_prompt,
            )

        except Exception as e:
            print("Error calling OpenAI server model:", e)


if __name__ == "__main__":
    main()
