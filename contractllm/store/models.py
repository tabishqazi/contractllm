from __future__ import annotations
import hashlib
import json
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field, computed_field


class ContractDefinition(BaseModel):
    """
    The full definition of a prompt contract.
    This is what gets stored when a contract is first declared.
    The schema_hash is computed from the content — not provided by the user.
    This prevents version string lies: if the content changes but version
    doesn't, we detect it. Same principle as Git's content addressing.
    """
    name: str = Field(description="Human-readable name, e.g. 'summarise_article'")
    version: str = Field(description="Version string, e.g. 'v1', 'v2'")
    system_prompt: str = Field(description="The system prompt sent to the LLM")
    input_schema: dict[str, Any] = Field(
        description="JSON Schema of the expected input, generated from Pydantic"
    )
    output_schema: dict[str, Any] = Field(
        description="JSON Schema of the expected output, generated from Pydantic"
    )
    provider: str = Field(description="Provider name: 'openai' or 'anthropic'")
    model: str = Field(description="Model identifier, e.g. 'gpt-4o-mini'")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def schema_hash(self) -> str:
        """
        A SHA-256 hash of the contract's content (not the version string).
        If you change the system prompt or any schema field, this hash changes.
        If the hash changes but version doesn't, the system warns you.
        This is the ground truth for 'did anything actually change?'
        """
        content = {
            "system_prompt": self.system_prompt,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "provider": self.provider,
            "model": self.model,
        }
        serialised = json.dumps(content, sort_keys=True)
        return hashlib.sha256(serialised.encode()).hexdigest()[:16]


class ContractRun(BaseModel):
    """
    A single execution of a contract — one LLM call.
    Stored after every run for regression comparison.
    """
    contract_name: str
    contract_version: str
    schema_hash: str
    input_data: dict[str, Any] = Field(description="The actual input values")
    raw_output: str = Field(description="The raw string the LLM returned")
    parsed_output: dict[str, Any] = Field(description="The validated parsed output")
    retry_count: int = Field(default=0, description="How many retries were needed")
    latency_ms: int = Field(description="Total latency including retries")
    tokens_used: int = Field(description="Total tokens across all attempts")
    provider: str
    model: str
    succeeded: bool
    error: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RegressionReport(BaseModel):
    """
    The output of comparing two contract versions.
    Tells you: did the outputs change? Did the schema change?
    """
    contract_name: str
    version_a: str
    version_b: str
    schema_changed: bool
    schema_hash_a: str
    schema_hash_b: str
    sample_count: int = Field(description="How many runs were compared")
    output_similarity: float = Field(
        description="Average semantic similarity 0.0-1.0 across compared runs"
    )
    regressions_detected: list[str] = Field(
        description="List of human-readable regression descriptions"
    )
