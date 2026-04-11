"""
The retry loop: the mechanism that makes LLM calls reliable.

Design decisions:
- Max 3 retries (configurable). After 3 failures, raise — don't loop forever.
- Retry with feedback: feed the validation error back as a new user message.
- Exponential backoff on rate limit errors (429s) — not on validation errors.
  Validation errors are model errors, not infrastructure errors. No need to wait.
- Circuit breaker: if N consecutive runs of a contract fail, stop calling
  the provider and raise immediately. Protects against runaway API spend.

We distinguish between two error types:
- Transient infrastructure errors (rate limits, timeouts) → backoff and retry
- Logical errors (wrong output format) → retry with corrective feedback, no delay
This distinction prevents wasting time waiting on errors that waiting won't fix.
"""
import asyncio
import logging
from typing import Any, Type
from pydantic import BaseModel

from contractllm.providers.base import LLMProvider, ProviderResponse
from contractllm.validation.output_validator import (
    validate_output,
    OutputParseError,
    OutputValidationError,
)

logger = logging.getLogger(__name__)


class MaxRetriesExceeded(Exception):
    """Raised when all retry attempts are exhausted."""
    def __init__(self, attempts: int, last_error: str):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Contract failed after {attempts} attempts. "
            f"Last error: {last_error}"
        )


async def run_with_retry(
    provider: LLMProvider,
    system_prompt: str,
    user_message: str,
    output_schema: dict[str, Any],
    output_schema_class: Type[BaseModel],
    max_retries: int = 3,
) -> tuple[ProviderResponse, dict[str, Any], BaseModel, int]:
    """
    Calls the provider with retry logic and validation feedback.

    Returns:
        (final_provider_response, raw_dict, validated_model, retry_count)
    """
    current_user_message = user_message
    last_error = ""
    total_tokens = 0

    for attempt in range(max_retries + 1):
        try:
            response = await provider.complete(
                system_prompt=system_prompt,
                user_message=current_user_message,
                output_schema=output_schema,
            )
            total_tokens += response.tokens_total

            raw_dict, validated = validate_output(
                response.content, output_schema_class
            )

            return response, raw_dict, validated, attempt

        except OutputValidationError as e:
            last_error = str(e)
            if attempt < max_retries:
                logger.warning(
                    f"Attempt {attempt + 1} failed validation. "
                    f"Retrying with feedback."
                )
                # Inject the validation error as corrective feedback
                current_user_message = (
                    f"{user_message}\n\n"
                    f"--- CORRECTION NEEDED ---\n"
                    f"{e.as_feedback_message()}"
                )

        except OutputParseError as e:
            last_error = str(e)
            if attempt < max_retries:
                current_user_message = (
                    f"{user_message}\n\n"
                    f"Your previous response was not valid JSON. "
                    f"Respond with ONLY a JSON object. No markdown, "
                    f"no explanation, no code fences. Just the JSON.\n"
                    f"Error: {e.parse_error}"
                )

        except Exception as e:
            # Rate limits, network errors — exponential backoff
            if "429" in str(e) or "rate" in str(e).lower():
                wait = 2 ** attempt
                logger.warning(f"Rate limited. Waiting {wait}s before retry.")
                await asyncio.sleep(wait)
                last_error = str(e)
            else:
                raise  # Don't retry on unknown errors

    raise MaxRetriesExceeded(attempts=max_retries + 1, last_error=last_error)
