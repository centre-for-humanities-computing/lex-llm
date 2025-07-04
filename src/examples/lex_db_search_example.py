from lex_db_api.configuration import Configuration  # type: ignore
from lex_db_api.api.lex_db_api import LexDbApi  # type: ignore
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
        tables = api.get_tables()
        print("Tables:", tables)
    except Exception as e:
        print("Error calling GET /api/tables:", e)

    print("\nTesting full text search with GET /api/articles ...")
    try:
        results_fts = api.get_articles(query="Rundetårn", limit=2)
        print("Results:", results_fts)
    except Exception as e:
        print("Error calling GET /api/articles:", e)

    print("\nTesting /api/vector-search ...")
    try:
        req_vector = VectorSearchRequest(
            query_text="Hvad er Rundetårn?",
            top_k=3,
        )
        results_vector = api.vector_search("small_003", req_vector)
        print("VectorSearchResults:", results_vector)
        if results_vector.results:
            article_ids = {
                int(result["source_article_id"]) for result in results_vector.results
            }
            print(article_ids)
            print(type(article_ids))
            full_articles = api.get_articles(ids=str(list(article_ids)))
            for article in full_articles:
                print(article)

    except Exception as e:
        print("Error calling /api/vector-search:", e)


if __name__ == "__main__":
    main()
