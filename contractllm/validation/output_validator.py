"""
Output validation: parse the LLM's raw string response and validate
it against the contract's output schema.

Two failure modes:
1. JSON parse failure — the LLM didn't return valid JSON at all
2. Schema validation failure — valid JSON but wrong shape/types

Both failures trigger the retry loop with different error messages.
The error message content matters — it becomes the feedback the LLM
receives in the next attempt.
"""
import json
from typing import Any, Type
from pydantic import BaseModel, ValidationError


class OutputParseError(Exception):
    """LLM returned something that isn't valid JSON."""
    def __init__(self, raw_output: str, parse_error: str):
        self.raw_output = raw_output
        self.parse_error = parse_error
        super().__init__(
            f"LLM response is not valid JSON.\n"
            f"Parse error: {parse_error}\n"
            f"Raw output (first 200 chars): {raw_output[:200]}"
        )


class OutputValidationError(Exception):
    """LLM returned valid JSON but wrong shape."""
    def __init__(self, data: dict, pydantic_errors: list[dict]):
        self.data = data
        self.pydantic_errors = pydantic_errors
        field_errors = [
            f"  - {' -> '.join(str(loc) for loc in err['loc'])}: {err['msg']}"
            for err in pydantic_errors
        ]
        super().__init__(
            "LLM output failed schema validation:\n" + "\n".join(field_errors)
        )

    def as_feedback_message(self) -> str:
        """
        Formats the validation error as a user message to send back
        to the LLM in the retry loop.

        This is the key innovation: we don't just say 'try again'.
        We tell the model exactly what was wrong. Like a compiler
        error message — specific, actionable, pointing to the problem.
        """
        return (
            f"Your previous response failed validation. "
            f"Please fix these issues and respond again with valid JSON:\n"
            f"{str(self)}\n\n"
            f"Your previous response was:\n{json.dumps(self.data, indent=2)}"
        )


def validate_output(
    raw_output: str,
    schema_class: Type[BaseModel],
) -> tuple[dict[str, Any], BaseModel]:
    """
    Parse and validate LLM output.
    Returns (raw_dict, validated_model) on success.
    Raises OutputParseError or OutputValidationError on failure.
    """
    # Strip markdown code fences if the model wrapped its JSON
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1])

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise OutputParseError(raw_output, str(e)) from e

    try:
        validated = schema_class.model_validate(data)
        return data, validated
    except ValidationError as e:
        raise OutputValidationError(data, e.errors(include_url=False)) from e
