"""
The @contract decorator — the public-facing API of llm-contracts.

Usage:
    from contractllm import contract, OpenAIProvider
    from pydantic import BaseModel

    class ArticleInput(BaseModel):
        text: str
        max_words: int = 100

    class SummaryOutput(BaseModel):
        summary: str
        key_points: list[str]
        word_count: int

    provider = OpenAIProvider(model="gpt-4o-mini")

    @contract(
        name="summarise_article",
        version="v1",
        system_prompt="You are a precise summarisation assistant.",
        input_schema=ArticleInput,
        output_schema=SummaryOutput,
        provider=provider,
    )
    async def summarise(input_data: ArticleInput) -> SummaryOutput:
        # The body of this function is never executed.
        # The decorator replaces it entirely.
        # This is the decorator pattern: the original function is
        # just a typed placeholder for the decorator to wrap.
        ...

Design choice: why does the function body get replaced?
Because the function signature (ArticleInput → SummaryOutput) is the
contract. The implementation (call LLM, validate, retry) is always the
same regardless of what the user puts in the body. So the body is
irrelevant — we use it only for its type annotations.
"""
import functools
import time
from typing import Any, Callable, Type
from pydantic import BaseModel

from contractllm.providers.base import LLMProvider
from contractllm.validation.input_validator import validate_input
from contractllm.retry.loop import run_with_retry
from contractllm.store.version_store import VersionStore
from contractllm.store.models import ContractDefinition, ContractRun


def contract(
    name: str,
    version: str,
    system_prompt: str,
    input_schema: Type[BaseModel],
    output_schema: Type[BaseModel],
    provider: LLMProvider,
    max_retries: int = 3,
    store: VersionStore | None = None,
):
    """
    Decorator factory. Returns a decorator that wraps an async function
    with full contract enforcement.

    Why a factory (decorator that returns a decorator)?
    Because we need to accept configuration parameters (name, version, etc.)
    A plain decorator takes only the function. A factory takes parameters
    and returns the actual decorator.

    The call stack: @contract(...) → contract_decorator → wrapper
    """
    def contract_decorator(func: Callable) -> Callable:

        # Compute and register the contract definition once at decoration time
        # (when the module loads), not on every call.
        output_schema_dict = output_schema.model_json_schema()
        input_schema_dict = input_schema.model_json_schema()

        definition = ContractDefinition(
            name=name,
            version=version,
            system_prompt=system_prompt,
            input_schema=input_schema_dict,
            output_schema=output_schema_dict,
            provider=provider.get_provider_name(),
            model=provider.model,
        )

        # Register with the version store if provided
        _store = store or VersionStore()
        _store.register_contract(definition)

        @functools.wraps(func)
        async def wrapper(input_data: dict[str, Any] | BaseModel) -> BaseModel:
            """
            The actual function that runs on every call.
            Replaces the original function body entirely.
            """
            start_time = time.monotonic()

            # Step 1: Validate input
            if isinstance(input_data, dict):
                validated_input = validate_input(input_data, input_schema)
            else:
                validated_input = input_data

            # Step 2: Build user message from validated input
            user_message = validated_input.model_dump_json(indent=2)

            # Step 3: Call provider with retry loop
            response, raw_dict, validated_output, retry_count = (
                await run_with_retry(
                    provider=provider,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    output_schema=output_schema_dict,
                    output_schema_class=output_schema,
                    max_retries=max_retries,
                )
            )

            latency_ms = int((time.monotonic() - start_time) * 1000)

            # Step 4: Store the run for regression comparison
            run = ContractRun(
                contract_name=name,
                contract_version=version,
                schema_hash=definition.schema_hash,
                input_data=validated_input.model_dump(),
                raw_output=response.content,
                parsed_output=raw_dict,
                retry_count=retry_count,
                latency_ms=latency_ms,
                tokens_used=response.tokens_total,
                provider=provider.get_provider_name(),
                model=response.model,
                succeeded=True,
            )
            _store.save_run(run)

            return validated_output

        # Attach contract metadata to the wrapper for introspection
        wrapper.contract_definition = definition
        wrapper.contract_version = version
        wrapper.contract_name = name

        return wrapper

    return contract_decorator
