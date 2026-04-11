"""
Input validation: check that what the caller passes matches the
contract's input schema BEFORE making an LLM call.

Why validate input? Because an invalid input will produce a bad
output that then fails output validation — wasting a full LLM call.
Fail fast at the boundary is cheaper than fail late inside the system.

This is the same principle as Zod validation on API routes in your
Node.js projects — validate at the edge before touching business logic.
"""
from typing import Any, Type
from pydantic import BaseModel, ValidationError


class InputValidationError(Exception):
    """
    Raised when input doesn't match the contract's input schema.
    Includes the Pydantic error details so the caller knows exactly
    what field failed and why.
    """
    def __init__(self, message: str, pydantic_errors: list[dict]):
        super().__init__(message)
        self.pydantic_errors = pydantic_errors


def validate_input(
    data: dict[str, Any],
    schema_class: Type[BaseModel],
) -> BaseModel:
    """
    Validates input data against a Pydantic model.
    Returns the validated model instance (with coercion applied).
    Raises InputValidationError with detailed field errors if invalid.
    """
    try:
        return schema_class.model_validate(data)
    except ValidationError as e:
        errors = e.errors(include_url=False)
        field_errors = [
            f"  - {' -> '.join(str(loc) for loc in err['loc'])}: {err['msg']}"
            for err in errors
        ]
        message = (
            f"Input validation failed for {schema_class.__name__}:\n"
            + "\n".join(field_errors)
        )
        raise InputValidationError(message, errors) from e
