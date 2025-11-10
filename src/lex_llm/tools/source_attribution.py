"""Source attribution utilities using LLM analysis."""

import json
from typing import List
from ..api.connectors.lex_db_connector import LexArticle
from ..api.connectors.openai_provider import LLMProvider


async def extract_used_sources_via_llm(
    response: str,
    retrieved_docs: List[LexArticle],
    llm_provider: LLMProvider,
) -> List[LexArticle]:
    """
    Uses an LLM to analyze the generated response and return only the Document objects
    that were actually used in generating the answer.

    Args:
        response: The generated response text
        retrieved_docs: List of all retrieved documents
        llm_provider: The LLM provider to use for attribution analysis
        deferral_message: The deferral message to check against

    Returns:
        List of LexArticle objects that were actually used in the response
    """
    if not response.strip():
        return []  # No sources used if deferral message was returned

    source_descriptions = "\n".join(
        [
            f"ID: {doc.id} | Title: {doc.title} | Content: {doc.text}"
            for doc in retrieved_docs
        ]
    )

    attribution_prompt = f"""
Analyze the assistant's response below and determine which of the provided sources were actually used to generate the answer.

Return ONLY a JSON-formatted list of source IDs that are directly referenced or used in the response.
If no sources were used, return an empty list.
    
Do not include explanations or markdown formatting.

## Sources
{source_descriptions}

## Assistant Response
{response}

## Expected Output Format
["id1", "id2", ...]
"""
    messages = [
        {
            "role": "system",
            "content": "You are a careful analyst who identifies which sources were used in a response from a chatbot.",
        },
        {"role": "user", "content": attribution_prompt},
    ]
    try:
        attribution_result = await llm_provider.generate(messages)  # type: ignore
        attribution_result = attribution_result.strip()
        # Clean up common LLM quirks
        if attribution_result.startswith("```json"):
            attribution_result = attribution_result[7:].split("```")[0].strip()
        elif attribution_result.startswith("```"):
            attribution_result = attribution_result[3:].split("```")[0].strip()
        used_ids = [int(id) for id in json.loads(attribution_result)]
        used_sources = [doc for doc in retrieved_docs if doc.id in used_ids]
        return used_sources

    except (json.JSONDecodeError, Exception) as e:
        # Log error if possible, fallback to empty
        print(f"Failed to parse attribution result: {e}")
        return []
