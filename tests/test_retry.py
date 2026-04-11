import pytest
from unittest.mock import AsyncMock
from pydantic import BaseModel
from contractllm.providers.base import LLMProvider, ProviderResponse
from contractllm.retry.loop import run_with_retry, MaxRetriesExceeded
from contractllm.validation.output_validator import OutputValidationError


class TestOutput(BaseModel):
    result: str
    score: float


class FailingParseProvider(LLMProvider):
    """Provider that always returns invalid JSON."""
    def __init__(self):
        super().__init__(model="fail", temperature=0.0)

    async def complete(self, system_prompt, user_message, output_schema):
        return ProviderResponse(
            content="not json",
            tokens_input=5,
            tokens_output=5,
            model="fail",
            finish_reason="stop",
        )

    def get_provider_name(self) -> str:
        return "fail"


class FailingSchemaProvider(LLMProvider):
    """Provider that returns valid JSON but wrong schema."""
    def __init__(self):
        super().__init__(model="fail", temperature=0.0)

    async def complete(self, system_prompt, user_message, output_schema):
        return ProviderResponse(
            content='{"wrong": "value"}',
            tokens_input=5,
            tokens_output=5,
            model="fail",
            finish_reason="stop",
        )

    def get_provider_name(self) -> str:
        return "fail"


class SuccessProvider(LLMProvider):
    """Provider that returns correct output."""
    def __init__(self):
        super().__init__(model="ok", temperature=0.0)

    async def complete(self, system_prompt, user_message, output_schema):
        return ProviderResponse(
            content='{"result": "hello", "score": 0.9}',
            tokens_input=5,
            tokens_output=5,
            model="ok",
            finish_reason="stop",
        )

    def get_provider_name(self) -> str:
        return "ok"


@pytest.mark.asyncio
async def test_retry_succeeds_first_attempt():
    provider = SuccessProvider()
    response, raw, validated, retries = await run_with_retry(
        provider=provider,
        system_prompt="You are helpful.",
        user_message='{"query": "hello"}',
        output_schema=TestOutput.model_json_schema(),
        output_schema_class=TestOutput,
        max_retries=3,
    )
    assert retries == 0
    assert validated.result == "hello"
    assert validated.score == 0.9


@pytest.mark.asyncio
async def test_retry_exhausted_after_max_attempts():
    provider = FailingSchemaProvider()
    with pytest.raises(MaxRetriesExceeded) as exc_info:
        await run_with_retry(
            provider=provider,
            system_prompt="You are helpful.",
            user_message='{"query": "hello"}',
            output_schema=TestOutput.model_json_schema(),
            output_schema_class=TestOutput,
            max_retries=2,
        )
    assert exc_info.value.attempts == 3  # Initial + 2 retries


@pytest.mark.asyncio
async def test_retry_recovers_from_parse_failure():
    provider = FailingParseProvider()
    with pytest.raises(MaxRetriesExceeded) as exc_info:
        await run_with_retry(
            provider=provider,
            system_prompt="You are helpful.",
            user_message='{"query": "hello"}',
            output_schema=TestOutput.model_json_schema(),
            output_schema_class=TestOutput,
            max_retries=3,
        )
    assert "not valid JSON" in str(exc_info.value.last_error)
