"""
The LLMProvider abstraction.

Why an abstract base class (ABC) instead of a protocol or duck typing?

- ABC gives you a clear error at class definition time if you forget to
  implement a required method. Protocol and duck typing fail at call time.
- ABC is self-documenting — anyone reading the code sees exactly what
  interface a provider must implement.
- ABC allows shared utility methods (like _build_error_message) that
  concrete providers inherit without reimplementing.

Design principle: program against the interface, not the implementation.
Your application code never imports OpenAI or Anthropic directly — only
LLMProvider. This is the Dependency Inversion Principle.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderResponse:
    """
    A standardised response from any LLM provider.
    Every provider returns different shapes — we normalise them here.
    This is the adapter pattern: convert provider-specific output
    into a shape our system understands.
    """
    content: str          # The raw text content
    tokens_input: int     # Prompt tokens
    tokens_output: int    # Completion tokens
    model: str            # The model that actually responded
    finish_reason: str    # "stop", "length", "content_filter", etc.

    @property
    def tokens_total(self) -> int:
        return self.tokens_input + self.tokens_output


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    Concrete implementations: OpenAIProvider, AnthropicProvider.

    The interface is intentionally minimal — just what we need.
    Every method we add here becomes a burden on every provider implementation.
    """

    def __init__(self, model: str, temperature: float = 0.0):
        """
        Temperature 0.0 by default for deterministic output.
        For contract validation we want consistent results — lower temperature
        means more predictable structured data that's easier to validate.
        """
        self.model = model
        self.temperature = temperature

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        output_schema: dict[str, Any],
    ) -> ProviderResponse:
        """
        Send a message to the LLM and get a response.

        output_schema: JSON Schema dict passed to the provider to
        guide structured output. Providers implement this differently:
        - OpenAI: response_format with json_schema
        - Anthropic: schema injected into the system prompt
        """
        ...

    @abstractmethod
    def get_provider_name(self) -> str:
        """Returns 'openai', 'anthropic', etc. Used for logging and storage."""
        ...

    def _format_schema_for_prompt(self, schema: dict[str, Any]) -> str:
        """
        Utility method shared by providers that inject schema into prompts.
        Concrete providers can call this rather than reimplementing.
        """
        import json
        return f"Respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
