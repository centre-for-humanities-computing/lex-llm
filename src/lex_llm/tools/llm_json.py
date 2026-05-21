"""Shared utilities for LLM-based tools."""

import json
from typing import Any


def parse_json_response(raw: str) -> dict[str, Any]:
    """Parse a JSON response from an LLM, handling common formatting quirks.

    Handles:
    - Markdown code blocks (```json ... ``` or ``` ... ```)
    - Leading/trailing whitespace
    - Extra text before/after the JSON object

    Args:
        raw: The raw LLM response string.

    Returns:
        Parsed JSON as a dictionary.

    Raises:
        ValueError: If the response cannot be parsed as JSON.
    """
    text = raw.strip()

    # Strip markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
        if "```" in text:
            text = text[: text.index("```")]
    elif text.startswith("```"):
        text = text[3:]
        if "```" in text:
            text = text[: text.index("```")]

    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)  # type: ignore
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the text (handles extra text before/after)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])  # type: ignore
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Failed to parse LLM response as JSON: {raw[:200]}...")
