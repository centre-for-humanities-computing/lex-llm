"""Editorial workflow v2 with a caller-chosen LLM.

Same steps as editorial_workflow_v2_gemma4_26b, but the LLM endpoint comes
from the request (``request.llm``: base_url, api_key, model, extra_body), so
the frontend can test and evaluate any model on any OpenAI-compatible
provider. Requests without ``llm`` are rejected; there is no default model.

Steps:
1. interpret_and_route — Combine query interpretation + scope routing
2. generate_deferral — Generate deferral message if out of scope
3. hybrid_search — Single-stage hybrid search with no separate corrective retrieval stage
4. generate_lead_and_body — Single streaming call producing bold lead + body
5. generate_source_list — Source attribution for conversation history
"""

import hashlib
import json
from collections import OrderedDict

from fastapi import HTTPException

from lex_llm.api.connectors.generic_provider import GenericProvider
from lex_llm.tools import hybrid_search

from ..api.orchestrator import Orchestrator
from ..api.event_models import LLMConfig, WorkflowRunRequest
from ..tools import (
    interpret_and_route,
    generate_deferral,
    generate_lead_and_body_v2,
    generate_source_list_v2,
)
from ..prompts_search_synthesis import (
    get_lead_and_body_prompt_v2,
    _format_date as _format_date,
)
from datetime import date

# LRU cache of providers, so requests with the same LLM config share one
# client and its connection pool. Evicted providers are not closed
# explicitly: a request still streaming may hold one, and the openai SDK
# closes the client itself once nothing references it.
_MAX_PROVIDERS = 8
_llm_providers: "OrderedDict[str, GenericProvider]" = OrderedDict()


def _cache_key(llm: LLMConfig) -> str:
    """Hash of the full config, so the raw API key is not kept as a dict key."""
    raw = json.dumps(
        [
            llm.base_url,
            llm.api_key.get_secret_value(),
            llm.model,
            llm.extra_body,
        ],
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _get_provider(llm: LLMConfig) -> GenericProvider:
    """Return the cached provider for this config, creating it if needed."""
    key = _cache_key(llm)
    provider = _llm_providers.get(key)
    if provider is not None:
        _llm_providers.move_to_end(key)
        return provider
    provider = GenericProvider(
        base_url=llm.base_url,
        api_key=llm.api_key.get_secret_value(),
        model=llm.model,
        extra_body=llm.extra_body,
    )
    _llm_providers[key] = provider
    if len(_llm_providers) > _MAX_PROVIDERS:
        _llm_providers.popitem(last=False)
    return provider


def get_workflow(request: WorkflowRunRequest) -> Orchestrator:
    if request.llm is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "This workflow requires an 'llm' object with base_url, "
                "api_key and model."
            ),
        )
    _llm = _get_provider(request.llm)

    return Orchestrator(
        request=request,
        steps=[
            interpret_and_route(llm_provider=_llm),
            generate_deferral(llm_provider=_llm),
            hybrid_search(
                index_name="article_embeddings_e5",
                top_k=20,
                top_k_semantic=30,
                top_k_fts=30,
                rrf_k=60,
            ),
            generate_lead_and_body_v2(
                llm_provider=_llm,
                system_prompt=get_lead_and_body_prompt_v2(),
                current_date=_format_date(date.today()),
            ),
            generate_source_list_v2(llm_provider=_llm),
        ],
        context={"conversation_history": request.conversation_history},
        use_clean_history=True,
    )


def get_metadata() -> dict:
    return {
        "workflow_id": "editorial_workflow_v2_llm_configurable",
        "name": "Editorial workflow v2 with configurable LLM",
        "status": "active",
        "description": (
            "Same as editorial_workflow_v2_gemma4_26b, but the LLM is chosen "
            "per request: the request must include an 'llm' object with "
            "base_url, api_key, model and optional extra_body, pointing at any "
            "OpenAI-compatible provider. Intended for testing and evaluating "
            "models."
        ),
        "steps": [
            {
                "name": "Interpret & Route",
                "description": "Interprets the user query and determines if it's within scope. Returns lists of keywords and subqueries for retrieval.",
                "inputs": ["user_input"],
                "outputs": [
                    "query_interpretation",
                    "is_in_scope",
                    "routing_reason",
                    "keywords",
                    "subqueries",
                ],
            },
            {
                "name": "Generate Deferral",
                "description": "Generates a deferral message if the query is out of scope.",
                "inputs": ["is_in_scope", "routing_reason"],
                "outputs": ["final_response"],
            },
            {
                "name": "Hybrid Search",
                "description": (
                    "Single-stage hybrid search (semantic + full-text, RRF-fused) "
                    "using the keywords and subqueries from interpretation."
                ),
                "inputs": [
                    "user_input",
                    "query_interpretation",
                    "keywords",
                    "subqueries",
                ],
                "outputs": ["retrieved_docs", "insufficient_context"],
            },
            {
                "name": "Generate Lead & Body (merged)",
                "description": (
                    "Single streaming LLM call that produces a bold Markdown lead "
                    "paragraph followed by an elaborating answer body. Streams as "
                    "text_chunk events."
                ),
                "inputs": [
                    "retrieved_docs",
                    "retrieved_chunks",
                    "conversation_history",
                    "user_input",
                    "query_interpretation",
                ],
                "outputs": ["final_response", "answer_body", "lead_paragraph"],
            },
            {
                "name": "Generate Source List",
                "description": "Attributes which sources were used in the answer.",
                "inputs": ["answer_body", "retrieved_docs"],
                "outputs": ["used_sources", "system_prompt", "final_response"],
            },
        ],
        "author": "Zafar",
        "version": "2.0.0",
    }
