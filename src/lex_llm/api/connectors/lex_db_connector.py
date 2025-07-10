from typing import List
import httpx
import os
from pydantic import BaseModel

from lex_db_api.api.lex_db_api import LexDbApi
from lex_db_api.api_client import ApiClient
from lex_db_api.configuration import Configuration
from lex_db_api.models.vector_search_request import VectorSearchRequest

lexdb_client = ApiClient(
    configuration=Configuration(host=os.getenv("DB_HOST", "http://localhost:8000"))
)
lexdb_api = LexDbApi(api_client=lexdb_client)


class LexArticle(BaseModel):
    id: int
    title: str
    text: str
    url: str


class LexDBConnector:
    """Handles communication with the Lex DB service."""

    async def vector_search(
        self, query: str, top_k: int = 5, index_name: str = "small_003"
    ) -> List[LexArticle]:
        """Performs a vector search against the knowledge base."""

        try:
            vec_req = VectorSearchRequest(query_text=query, top_k=top_k)
            vector_search_result = lexdb_api.vector_search(index_name, vec_req)
            if vector_search_result.results:
                search_results = lexdb_api.get_articles(
                    ids=str(
                        list(
                            {
                                int(result["source_article_id"])
                                for result in vector_search_result.results
                            }
                        )
                    )
                )
                if search_results.entries:
                    return [
                        LexArticle(
                            id=result["id"],
                            title=result["title"],
                            text=result["xhtml_md"],
                            url=result["url"],
                        )
                        for result in search_results.entries
                    ]

            return []
        except httpx.RequestError as e:
            print(f"Error connecting to LexDB: {e}")
            # TODO: more robust error handling/logging
            return []
