from lex_db_api.configuration import Configuration  # type: ignore
from lex_db_api.api.lex_db_api import LexDbApi  # type: ignore
from lex_db_api.models.full_text_search_request import FullTextSearchRequest  # type: ignore
from lex_db_api.models.vector_search_request import VectorSearchRequest  # type: ignore
from lex_db_api.api_client import ApiClient  # type: ignore


def main() -> None:
    """
    Main function to test the LexDbApi endpoints for retrieving tables,
    performing full-text search, and vector search.
    """
    # Set up the API client to point to your running API
    import os

    api_host = os.getenv("DB_HOST", "http://0.0.0.0:8000")
    api_client = ApiClient(configuration=Configuration(host=api_host))
    api = LexDbApi(api_client=api_client)

    try:
        tables = api.get_tables_api_tables_get()
        print("Tables:", tables)
    except Exception as e:
        print("Error calling /api/tables:", e)

    print("\nTesting /api/search ...")
    try:
        req_fts = FullTextSearchRequest(query="Rundetårn", limit=2)
        results_fts = api.full_text_search_api_search_post(req_fts)
        print("FullTextSearchResults:", results_fts)
    except Exception as e:
        print("Error calling /api/search:", e)

    print("\nTesting /api/vector-search ...")
    try:
        req_vector = VectorSearchRequest(
            vector_index_name="small_003",
            query_text="Hvad er Rundetårn?",
            embedding_model_choice="openai_small_003",
            top_k=2,
        )
        results_vector = api.vector_search_api_vector_search_post(req_vector)
        print("VectorSearchResults:", results_vector)
    except Exception as e:
        print("Error calling /api/vector-search:", e)


if __name__ == "__main__":
    main()
