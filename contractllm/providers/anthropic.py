import os
from typing import Any

try:
    import anthropic as _anthropic
except ImportError:
    raise ImportError(
        "Anthropic provider requires the anthropic package.\n"
        "Install it with: pip install llm-contracts[anthropic]\n"
        "Or: pip install anthropic"
    ) from None

from .base import LLMProvider, ProviderResponse


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude implementation. Requires: pip install llm-contracts[anthropic]
    """

    def __init__(
        self,
        model: str = "claude-3-5-haiku-20241022",
        temperature: float = 0.0,
    ):
        super().__init__(model=model, temperature=temperature)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Add it to your .env file: ANTHROPIC_API_KEY=sk-ant-..."
            )
        self._client = _anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        output_schema: dict[str, Any],
    ) -> ProviderResponse:
        schema_instruction = self._format_schema_for_prompt(output_schema)
        full_system = f"{system_prompt}\n\n{schema_instruction}"

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=self.temperature,
            system=full_system,
            messages=[{"role": "user", "content": user_message}],
        )
        content_block = response.content[0]
        content = content_block.text if hasattr(content_block, "text") else ""

        return ProviderResponse(
            content=content,
            tokens_input=response.usage.input_tokens,
            tokens_output=response.usage.output_tokens,
            model=self.model,
            finish_reason=response.stop_reason or "stop",
        )

    def get_provider_name(self) -> str:
        return "anthropic"
