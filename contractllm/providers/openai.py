import os
from typing import Any

try:
    import openai as _openai
except ImportError:
    raise ImportError(
        "OpenAI provider requires the openai package.\n"
        "Install it with: pip install llm-contracts[openai]\n"
        "Or: pip install openai"
    ) from None

from .base import LLMProvider, ProviderResponse


class OpenAIProvider(LLMProvider):
    """
    OpenAI implementation. Requires: pip install llm-contracts[openai]
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        super().__init__(model=model, temperature=temperature)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Add it to your .env file: OPENAI_API_KEY=sk-..."
            )
        self._client = _openai.AsyncOpenAI(api_key=api_key)

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        output_schema: dict[str, Any],
    ) -> ProviderResponse:
        # OpenAI requires additionalProperties: false on object schemas
        schema = dict(output_schema)
        if schema.get("type") == "object" and "properties" in schema:
            schema["additionalProperties"] = False

        response = await self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "contract_output",
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        choice = response.choices[0]
        return ProviderResponse(
            content=choice.message.content or "",
            tokens_input=response.usage.prompt_tokens,
            tokens_output=response.usage.completion_tokens,
            model=response.model,
            finish_reason=choice.finish_reason,
        )

    def get_provider_name(self) -> str:
        return "openai"
