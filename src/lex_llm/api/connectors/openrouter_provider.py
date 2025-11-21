from typing import AsyncGenerator, List, Optional
import litellm
from ..event_models import ConversationMessage
from .openai_provider import LLMProvider


class OpenRouterProvider(LLMProvider):
    """Implementation for OpenRouter's API with configurable model and provider."""

    def __init__(
        self,
        model: str = "google/gemma-3-27b-it",
        providers: Optional[List[str]] = ["nebius/fp8"],
    ):
        """
        Initialize OpenRouter provider.

        Args:
            model: The model to use (default: google/gemma-3-27b-it)
            provider: The provider preference for routing (default: nebius/fp8)
                     Set to None to use OpenRouter's default routing
            api_key: OpenRouter API key (default: reads from OPENROUTER_API_KEY env var)
        """
        self.model = model
        self.providers = providers

    def _merge_consecutive_messages(
        self, messages: List[ConversationMessage]
    ) -> List[ConversationMessage]:
        """Merge consecutive messages with the same role."""
        if not messages:
            return messages

        # Convert first message to dict and create a copy
        merged = [dict(messages[0]) if isinstance(messages[0], dict) else messages[0].model_dump()]

        for msg in messages[1:]:
            # Convert to dict if it's a Pydantic model
            msg_dict = dict(msg) if isinstance(msg, dict) else msg.model_dump()

            if msg_dict["role"] == merged[-1]["role"]:
                # Merge content with the previous message
                merged[-1]["content"] = f"{merged[-1]['content']}\n\n{msg_dict['content']}"
            else:
                merged.append(msg_dict)

        return merged  # type: ignore

    async def generate_stream(
        self, messages: List[ConversationMessage]
    ) -> AsyncGenerator[str, None]:
        """Calls the OpenRouter API and streams the response."""
        # Merge consecutive messages with the same role
        merged_messages = self._merge_consecutive_messages(messages)

        # Prepare extra headers for provider routing
        extra_body = {}
        if self.providers:
            extra_body["provider"] = {"order": self.providers, "allow_fallbacks": False}

        stream = await litellm.acompletion(
            model=f"openrouter/{self.model}",
            messages=merged_messages,
            stream=True,
            extra_body=extra_body if extra_body else None,
        )
        async for chunk in stream:  # type: ignore
            content = chunk.choices[0].delta.content
            if content:
                yield content

    async def generate(self, messages: List[ConversationMessage]) -> str:
        """Generates a response as a single text chunk."""
        response = ""
        async for chunk in self.generate_stream(messages):
            response += chunk
        return response
