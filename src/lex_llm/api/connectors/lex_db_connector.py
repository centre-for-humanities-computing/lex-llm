from typing import List
import httpx
import os
from pydantic import BaseModel

from lex_db_api.api.lex_db_api import LexDbApi
from lex_db_api.api_client import ApiClient
from lex_db_api.configuration import Configuration
from lex_db_api.models.vector_search_request import VectorSearchRequest
from lex_db_api.models.hybrid_search_request import HybridSearchRequest
from lex_db_api.models.hy_de_search_request import HyDESearchRequest
from lex_db_api.models.hybrid_hy_de_search_request import HybridHyDESearchRequest


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
                                int(result.source_article_id)
                                for result in vector_search_result.results
                            }
                        )
                    )
                )
                if search_results.entries:
                    return [
                        LexArticle(
                            id=result.id,
                            title=result.title,
                            text=result.xhtml_md,
                            url=result.url,
                        )
                        for result in search_results.entries
                    ]

            return []
        except httpx.RequestError as e:
            print(f"Error connecting to LexDB: {e}")
            # TODO: more robust error handling/logging
            return []

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        top_k_semantic: int = 50,
        top_k_fts: int = 50,
        rrf_k: int = 60,
        index_name: str = "article_embeddings_e5",
    ) -> List[LexArticle]:
        """Performs hybrid search using RRF fusion via the lex-db API."""

        try:
            hybrid_req = HybridSearchRequest(
                query_text=query,
                top_k=top_k,
                top_k_semantic=top_k_semantic,
                top_k_fts=top_k_fts,
                rrf_k=rrf_k,
            )

            hybrid_search_result = lexdb_api.hybrid_search(index_name, hybrid_req)

            if hybrid_search_result.results:
                return [
                    LexArticle(
                        id=result.article_id,
                        title=result.article_headword,
                        text=result.chunk_text,
                        url=f"https://lex.dk/{result.article_headword}",
                    )
                    for result in hybrid_search_result.results
                ]

            return []
        except httpx.RequestError as e:
            print(f"Error connecting to LexDB: {e}")
            return []

    async def hyde_search(
        self,
        query: str,
        top_k: int = 10,
        index_name: str = "article_embeddings_e5",
    ) -> List[LexArticle]:
        """Performs HyDE search via the lex-db API."""

        try:
            hyde_req = HyDESearchRequest(query_text=query, top_k=top_k)

            hyde_search_result = lexdb_api.hyde_search(index_name, hyde_req)

            if hyde_search_result:
                return [
                    LexArticle(
                        id=result["article_id"],
                        title=result["headword"],
                        text=result["text"],
                        url=f"https://lex.dk/{result['headword']}",
                    )
                    for result in hyde_search_result
                ]

            return []
        except httpx.RequestError as e:
            print(f"Error connecting to LexDB: {e}")
            return []

    async def hybrid_hyde_search(
        self,
        query: str,
        top_k: int = 10,
        top_k_hyde: int = 50,
        top_k_fts: int = 50,
        rrf_k: int = 60,
        index_name: str = "article_embeddings_e5",
    ) -> List[LexArticle]:
        """Performs hybrid HyDE search with adaptive RRF weighting."""

        try:
            hybrid_hyde_req = HybridHyDESearchRequest(
                query_text=query,
                top_k=top_k,
                top_k_hyde=top_k_hyde,
                top_k_fts=top_k_fts,
                rrf_k=rrf_k,
            )

            hybrid_hyde_result = lexdb_api.hybrid_hyde_search(
                index_name, hybrid_hyde_req
            )

            if hybrid_hyde_result:
                return [
                    LexArticle(
                        id=result.article_id,
                        title=result.article_headword,
                        text=result.chunk_text,
                        url=f"https://lex.dk/{result.article_headword}",
                    )
                    for result in hybrid_hyde_result
                ]

            return []
        except httpx.RequestError as e:
            print(f"Error connecting to LexDB: {e}")
            return []
