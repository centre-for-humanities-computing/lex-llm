import httpx
import os
from typing_extensions import deprecated
from pydantic import BaseModel

from lex_db_api.api.lex_db_api import LexDbApi
from lex_db_api.api_client import ApiClient
from lex_db_api.configuration import Configuration
from lex_db_api.models.search_method import SearchMethod
from lex_db_api.models.text_type import TextType
from lex_db_api.models.vector_search_request import VectorSearchRequest
from lex_db_api.models.hybrid_search_request import HybridSearchRequest
from lex_db_api.models.batch_vector_search_request import BatchVectorSearchRequest
from lex_db_api.models.batch_fulltext_search_request import BatchFulltextSearchRequest

lexdb_client = ApiClient(
    configuration=Configuration(host=os.getenv("DB_HOST", "http://localhost:8000"))
)
lexdb_api = LexDbApi(api_client=lexdb_client)


class LexChunk(BaseModel):
    """A single chunk retrieved from the knowledge base.

    Chunks are the atomic unit returned by search endpoints. They preserve
    chunk-level granularity for RRF fusion, reranking, and sorted ordering
    in prompts (to maximize KV cache hits during generation).
    """

    article_id: int
    chunk_seq: int
    chunk_text: str
    title: str | None = None
    url: str | None = None


class LexArticle(BaseModel):
    """An article reconstructed from grouped chunks.

    This is a convenience view for downstream consumers that need
    article-level data (e.g., source attribution, conversation history).
    """

    id: int
    title: str
    text: str
    url: str | None = None


def group_chunks_to_articles(chunks: list[LexChunk]) -> list[LexArticle]:
    """Group chunks by article_id into LexArticle objects.

    Chunks within each article are sorted by chunk_seq to ensure
    correct text ordering. Articles are returned in the order of
    first appearance of their chunks.
    """
    from collections import OrderedDict

    grouped: OrderedDict[int, list[LexChunk]] = OrderedDict()
    for chunk in chunks:
        grouped.setdefault(chunk.article_id, []).append(chunk)

    articles: list[LexArticle] = []
    for aid, article_chunks in grouped.items():
        # Sort chunks by sequence number
        article_chunks.sort(key=lambda c: c.chunk_seq)
        # Use title/url from the first chunk that has them
        title = next((c.title for c in article_chunks if c.title), "")
        url = next((c.url for c in article_chunks if c.url), None)
        articles.append(
            LexArticle(
                id=aid,
                title=title,
                text="\n\n".join(c.chunk_text for c in article_chunks),
                url=url,
            )
        )
    return articles


class LexDBConnector:
    """Handles communication with the Lex DB service."""

    async def vector_search(
        self, query: str, top_k: int = 5, index_name: str = "small_003"
    ) -> list[LexChunk]:
        """Performs a vector search against the knowledge base."""

        try:
            vec_req = VectorSearchRequest(query_text=query, top_k=top_k)
            vector_search_result = lexdb_api.vector_search(index_name, vec_req)
            if vector_search_result.results:
                return [
                    LexChunk(
                        article_id=int(result.source_article_id),
                        chunk_seq=result.chunk_seq,
                        chunk_text=result.chunk_text,
                        title=result.title,
                        url=result.url,
                    )
                    for result in vector_search_result.results
                ]

            return []
        except httpx.RequestError as e:
            print(f"Error connecting to LexDB: {e}")
            # TODO: more robust error handling/logging
            return []

    @deprecated(
        "Orchestrate hybrid search as a separate step instead of within the connector"
    )
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        top_k_semantic: int = 50,
        top_k_fts: int = 50,
        rrf_k: int = 60,
        index_name: str = "article_embeddings_e5",
        methods: list[SearchMethod] | None = None,
    ) -> list[LexChunk]:
        """Performs hybrid search using RRF fusion via the lex-db API."""
        try:
            hybrid_req = HybridSearchRequest(
                query_text=query,
                top_k=top_k,
                top_k_semantic=top_k_semantic,
                top_k_fts=top_k_fts,
                rrf_k=rrf_k,
                methods=methods,
            )

            hybrid_search_result = lexdb_api.hybrid_search(index_name, hybrid_req)

            if hybrid_search_result.results:
                return [
                    LexChunk(
                        article_id=int(result.article_id),
                        chunk_seq=result.chunk_sequence,
                        chunk_text=result.chunk_text,
                        title=result.title,
                        url=result.url,
                    )
                    for result in hybrid_search_result.results
                ]

            return []
        except httpx.RequestError as e:
            print(f"Error connecting to LexDB: {e}")
            return []

    @deprecated(
        "Orchestrate HyDE search as a separate step instead of within the connector"
    )
    async def hyde_search(
        self, query: str, top_k: int = 5, index_name: str = "article_embeddings_e5"
    ) -> list[LexChunk]:
        """Performs HyDE (Hypothetical Document Embeddings) search against the knowledge base."""

        try:
            hyde_req = VectorSearchRequest(query_text=query, top_k=top_k)
            hyde_search_result = lexdb_api.hyde_search(index_name, hyde_req)
            if hyde_search_result.results:
                return [
                    LexChunk(
                        article_id=int(result.source_article_id),
                        chunk_seq=result.chunk_seq,
                        chunk_text=result.chunk_text,
                        title=result.title,
                        url=result.url,
                    )
                    for result in hyde_search_result.results
                ]

            return []
        except httpx.RequestError as e:
            print(f"Error connecting to LexDB: {e}")
            return []

    async def batch_vector_search(
        self,
        queries: list[tuple[str, TextType]],
        top_k: int = 5,
        index_name: str = "article_embeddings_e5",
    ) -> list[list[LexChunk]]:
        """Performs batch vector search with multiple query texts.

        Each query is a (query_text, TextType) tuple, allowing callers to
        choose the appropriate text type per query (e.g. TextType.QUERY for
        short queries, TextType.PASSAGE for HyDE-style hypothetical documents).
        Results are returned as a list of lists — one inner list per query —
        preserving per-query ranking for downstream RRF fusion.
        """
        try:
            # BatchVectorSearchRequest expects queries as list of [query_text, TextType] pairs
            query_pairs: list[list[str]] = [[text, tt.value] for text, tt in queries]
            batch_req = BatchVectorSearchRequest(queries=query_pairs, top_k=top_k)
            batch_results = lexdb_api.batch_vector_search(index_name, batch_req)

            # batch_results is a list of VectorSearchResults (one per query)
            per_query_chunks: list[list[LexChunk]] = []
            for search_results in batch_results:
                query_chunks: list[LexChunk] = []
                if search_results.results:
                    for result in search_results.results:
                        query_chunks.append(
                            LexChunk(
                                article_id=int(result.source_article_id),
                                chunk_seq=result.chunk_seq,
                                chunk_text=result.chunk_text,
                                title=result.title,
                                url=result.url,
                            )
                        )
                per_query_chunks.append(query_chunks)

            return per_query_chunks
        except httpx.RequestError as e:
            print(f"Error connecting to LexDB: {e}")
            return [[] for _ in queries]

    async def batch_fulltext_search(
        self,
        queries: list[str],
        top_k: int = 50,
        index_name: str = "article_embeddings_e5",
    ) -> list[list[LexChunk]]:
        """Performs batch fulltext search with multiple keyword queries.

        Results are returned as a list of lists — one inner list per query —
        preserving per-query ranking for downstream RRF fusion.
        """
        try:
            batch_req = BatchFulltextSearchRequest(queries=queries, top_k=top_k)
            batch_results = lexdb_api.batch_fulltext_search(index_name, batch_req)

            # batch_results is a list of lists of RetrievalResult (one inner list per query)
            per_query_chunks: list[list[LexChunk]] = []
            for query_results in batch_results:
                query_chunks: list[LexChunk] = []
                for result in query_results:
                    query_chunks.append(
                        LexChunk(
                            article_id=int(result.article_id),
                            chunk_seq=result.chunk_sequence,
                            chunk_text=result.chunk_text,
                            title=result.title,
                            url=result.url,
                        )
                    )
                per_query_chunks.append(query_chunks)

            return per_query_chunks
        except httpx.RequestError as e:
            print(f"Error connecting to LexDB: {e}")
            return [[] for _ in queries]
