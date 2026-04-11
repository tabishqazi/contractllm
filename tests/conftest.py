import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel
from contractllm.providers.base import LLMProvider, ProviderResponse


class MockProvider(LLMProvider):
    """
    A mock provider for testing. Never makes real API calls.
    Interview point: mocking the provider lets us test the entire
    contract system — validation, retry, versioning — without
    spending money on API calls or needing network access in CI.
    This is why the adapter pattern matters: testability.
    """

    def __init__(self, responses: list[str]):
        super().__init__(model="mock-model", temperature=0.0)
        self._responses = iter(responses)
        self.call_count = 0

    async def complete(self, system_prompt, user_message, output_schema):
        self.call_count += 1
        content = next(self._responses)
        return ProviderResponse(
            content=content,
            tokens_input=10,
            tokens_output=20,
            model="mock-model",
            finish_reason="stop",
        )

    def get_provider_name(self) -> str:
        return "mock"


@pytest.fixture
def valid_json_provider():
    return MockProvider(['{"result": "hello", "score": 0.9}'])


@pytest.fixture
def retry_then_succeed_provider():
    return MockProvider([
        "not json at all",                          # Fails: parse error
        '{"wrong_field": "value"}',               # Fails: schema error
        '{"result": "hello", "score": 0.9}',        # Succeeds
    ])
