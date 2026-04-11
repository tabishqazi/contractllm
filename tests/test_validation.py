import pytest
from pydantic import BaseModel, Field
from contractllm.validation.input_validator import (
    validate_input,
    InputValidationError,
)
from contractllm.validation.output_validator import (
    validate_output,
    OutputParseError,
    OutputValidationError,
)


class ArticleInput(BaseModel):
    text: str = Field(min_length=1)
    max_words: int = Field(ge=1, le=1000)


class SummaryOutput(BaseModel):
    summary: str
    word_count: int


def test_validate_input_success():
    data = {"text": "Great product", "max_words": 50}
    validated = validate_input(data, ArticleInput)
    assert validated.text == "Great product"
    assert validated.max_words == 50


def test_validate_input_missing_field():
    data = {"text": "Great product"}  # missing max_words
    with pytest.raises(InputValidationError) as exc_info:
        validate_input(data, ArticleInput)
    assert "max_words" in str(exc_info.value)
    assert len(exc_info.value.pydantic_errors) > 0


def test_validate_input_wrong_type():
    data = {"text": "Great product", "max_words": "fifty"}
    with pytest.raises(InputValidationError):
        validate_input(data, ArticleInput)


def test_validate_output_success():
    raw = '{"summary": "Great article", "word_count": 50}'
    data, validated = validate_output(raw, SummaryOutput)
    assert data["summary"] == "Great article"
    assert validated.word_count == 50


def test_validate_output_strips_code_fences():
    raw = '```json\n{"summary": "Great article", "word_count": 50}\n```'
    data, validated = validate_output(raw, SummaryOutput)
    assert validated.summary == "Great article"


def test_validate_output_parse_failure():
    raw = "this is not json at all"
    with pytest.raises(OutputParseError) as exc_info:
        validate_output(raw, SummaryOutput)
    assert exc_info.value.raw_output == raw
    assert "not valid JSON" in str(exc_info.value)


def test_validate_output_schema_failure():
    raw = '{"wrong_field": "value", "word_count": 50}'
    with pytest.raises(OutputValidationError) as exc_info:
        validate_output(raw, SummaryOutput)
    assert "summary" in str(exc_info.value)  # Should mention missing field
    assert exc_info.value.data == {"wrong_field": "value", "word_count": 50}


def test_feedback_message_format():
    raw = '{"wrong_field": "value"}'
    with pytest.raises(OutputValidationError) as exc_info:
        validate_output(raw, SummaryOutput)

    feedback = exc_info.value.as_feedback_message()
    assert "failed validation" in feedback
    assert "summary" in feedback  # Should mention the missing field
    assert "wrong_field" in feedback  # Should mention what was wrong
