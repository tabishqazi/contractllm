import pytest
from pydantic import BaseModel
from contractllm.contract import contract


class TestInput(BaseModel):
    query: str


class TestOutput(BaseModel):
    result: str
    score: float


@pytest.mark.asyncio
async def test_successful_contract_call(valid_json_provider, tmp_path):
    from contractllm.store.version_store import VersionStore
    store = VersionStore(db_path=tmp_path / "test.db")

    @contract(
        name="test_contract",
        version="v1",
        system_prompt="You are helpful.",
        input_schema=TestInput,
        output_schema=TestOutput,
        provider=valid_json_provider,
        store=store,
    )
    async def my_func(data: TestInput) -> TestOutput: ...

    result = await my_func({"query": "hello"})

    assert isinstance(result, TestOutput)
    assert result.result == "hello"
    assert result.score == 0.9


@pytest.mark.asyncio
async def test_retries_on_bad_output(retry_then_succeed_provider, tmp_path):
    from contractllm.store.version_store import VersionStore
    store = VersionStore(db_path=tmp_path / "test.db")

    @contract(
        name="retry_test",
        version="v1",
        system_prompt="You are helpful.",
        input_schema=TestInput,
        output_schema=TestOutput,
        provider=retry_then_succeed_provider,
        store=store,
        max_retries=3,
    )
    async def my_func(data: TestInput) -> TestOutput: ...

    result = await my_func({"query": "hello"})
    assert isinstance(result, TestOutput)
    assert result.result == "hello"
    assert result.score == 0.9
