import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../build/lex-db-client')))

from lex_db_client.configuration import Configuration
from lex_db_client.api.lex_db_api import LexDbApi
from lex_db_client.models.full_text_search_request import FullTextSearchRequest
from lex_db_client.models.vector_search_request import VectorSearchRequest
from lex_db_client.api_client import ApiClient

def main():
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
        req = FullTextSearchRequest(query="Rundetårn", limit=2)
        results = api.full_text_search_api_search_post(req)
        print("FullTextSearchResults:", results)
    except Exception as e:
        print("Error calling /api/search:", e)

    print("\nTesting /api/vector-search ...")
    try:
        req = VectorSearchRequest(
            vector_index_name="small_003",
            query_text="Hvad er Rundetårn?",
            embedding_model_choice="openai_small_003",
            top_k=2
        )
        results = api.vector_search_api_vector_search_post(req)
        print("VectorSearchResults:", results)
    except Exception as e:
        print("Error calling /api/vector-search:", e)

if __name__ == "__main__":
    main()