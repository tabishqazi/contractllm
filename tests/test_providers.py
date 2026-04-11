import pytest
from pydantic import BaseModel
from contractllm.providers.base import LLMProvider, ProviderResponse
from contractllm.providers.openai import OpenAIProvider
from contractllm.providers.anthropic import AnthropicProvider


class TestOutput(BaseModel):
    result: str
    score: float


class TestProvider(LLMProvider):
    """Concrete implementation for testing the abstract interface."""

    async def complete(self, system_prompt, user_message, output_schema):
        return ProviderResponse(
            content='{"result": "ok", "score": 1.0}',
            tokens_input=5,
            tokens_output=10,
            model="test",
            finish_reason="stop",
        )

    def get_provider_name(self) -> str:
        return "test"


def test_provider_response_tokens():
    response = ProviderResponse(
        content="hello",
        tokens_input=10,
        tokens_output=20,
        model="test",
        finish_reason="stop",
    )
    assert response.tokens_total == 30


def test_provider_interface_contract():
    """Verify TestProvider satisfies LLMProvider interface."""
    provider = TestProvider(model="test-model")
    assert provider.model == "test-model"
    assert provider.temperature == 0.0
    assert provider.get_provider_name() == "test"


@pytest.mark.asyncio
async def test_provider_complete_returns_response():
    provider = TestProvider(model="test-model")
    response = await provider.complete(
        system_prompt="You are helpful.",
        user_message="hello",
        output_schema=TestOutput.model_json_schema(),
    )
    assert response.content == '{"result": "ok", "score": 1.0}'
    assert response.tokens_total == 15
    assert response.finish_reason == "stop"


def test_openai_provider_requires_api_key(monkeypatch):
    """OpenAIProvider raises ValueError if API key is missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIProvider()


def test_anthropic_provider_requires_api_key(monkeypatch):
    """AnthropicProvider raises ValueError if API key is missing."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider()
