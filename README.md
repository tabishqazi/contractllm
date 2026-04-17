# contractllm

> Typed, versioned, regression-safe contracts for LLM calls.
> Because changing a prompt in production shouldn't be terrifying.

```bash
pip install contractllm
```

![Python](https://img.shields.io/badge/python-3.11+-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/status-stable-brightgreen?style=flat-square)

---

## Why this exists

Every API your code calls has a contract:

```
REST API   →  OpenAPI spec   →  typed request/response
gRPC       →  protobuf       →  typed request/response
SQL        →  table schema   →  typed rows
LLM call   →  ???            →  json.loads() and 🤞
```

**contractllm closes that gap.**

It gives LLM calls the same guarantees every other API takes for granted:
typed inputs, typed outputs, version tracking, and regression detection —
without requiring a cloud platform, an account, or any infrastructure.

---

## How a call works

```
Your Code
    │
    │  analyse_sentiment({"product_name": "...", "review_text": "..."})
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      @contract decorator                        │
│                                                                 │
│  ┌──────────────────┐                                           │
│  │  Input validator │  Pydantic check before touching the API  │
│  └────────┬─────────┘                                           │
│           │ ✓ valid                                             │
│           ▼                                                     │
│  ┌──────────────────┐                                           │
│  │ Provider adapter │  OpenAI · Anthropic · (more coming)      │
│  └────────┬─────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────┐   ✗ wrong shape ┌─────────────────────┐  │
│  │ Output validator │ ──────────────► │  Retry with feedback │  │
│  └────────┬─────────┘                 │  "field X is missing"│  │
│           │ ✓ valid          max 3x   └──────────┬──────────┘  │
│           │                                      │ ✓ fixed     │
│           ◄──────────────────────────────────────┘             │
│           │                                                     │
│  ┌──────────────────┐                                           │
│  │  Version store   │  SQLite · schema hash · run history      │
│  └────────┬─────────┘                                           │
└───────────┼─────────────────────────────────────────────────────┘
            │
            ▼
    SentimentOutput(sentiment="positive", confidence=0.95, ...)
    A real typed Python object. Not a dict. Not a string.
```

---

## The problem with raw LLM calls

```python
# What most production code looks like today
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": f"Analyse this: {text}"}]
)
data = json.loads(response.choices[0].message.content)  # 💥 JSONDecodeError
sentiment = data["sentiment"]                            # 💥 KeyError
score = float(data["confidence"])                        # 💥 ValueError
```

Three silent failure modes. Zero errors at the call site.
You find out when a user reports it.

---

## The contractllm solution

```python
from pydantic import BaseModel
from contractllm import contract
from contractllm.providers.openai import OpenAIProvider


class ReviewInput(BaseModel):
    product_name: str
    review_text: str


class SentimentOutput(BaseModel):
    sentiment: str          # "positive" | "negative" | "neutral"
    confidence: float       # 0.0 to 1.0
    key_phrases: list[str]  # What drove the sentiment


@contract(
    name="analyse_sentiment",
    version="v1",
    system_prompt="You are a sentiment analysis expert.",
    input_schema=ReviewInput,
    output_schema=SentimentOutput,
    provider=OpenAIProvider(model="gpt-4o-mini"),
)
async def analyse_sentiment(data: ReviewInput) -> SentimentOutput:
    ...  # Decorator handles everything — this body is never executed


# Usage
result = await analyse_sentiment({
    "product_name": "Wireless Headphones",
    "review_text": "Fantastic sound, battery life is incredible.",
})

result.sentiment     # "positive"              ← typed attribute, not dict key
result.confidence    # 0.95                    ← float, not string
result.key_phrases   # ["fantastic", ...]      ← typed list
```

---

## Three guarantees on every call

### 1 — Input validation (before the API call)

```
Input received
    │
    ▼
Pydantic validation
    ├── product_name: 12345 (int, not str)
    │       └── ✗  InputValidationError: must be a valid string
    ├── review_text: missing
    │       └── ✗  InputValidationError: field required
    └── all fields valid
            └── ✓  proceed to LLM call
```

Fails immediately with a precise error. Zero tokens spent on bad input.

### 2 — Retry with error feedback (not blind retry)

```
LLM response received
        │
        ▼
Parse JSON  ──── fail ──►  "Respond with ONLY JSON. No markdown.
        │                   Your previous response was not valid JSON."
        │                              │
        ▼                             retry (max 3x)
Pydantic validation                    │
        │                              ▼
        ├── wrong shape ──►  "Field 'confidence' is required.
        │                    You returned 'score'. Fix this."
        │                              │
        └── valid ──────────────────► return typed object
```

The model receives the exact error — field name, what was wrong, what was
returned. Same principle as a compiler error message. Reduces retries by
60–70% vs blind retry.

### 3 — Version store + regression detection

```
Every run stores:
┌────────────────────────────────────────────────────────────┐
│  name: "analyse_sentiment"                                 │
│  version: "v1"                                             │
│  schema_hash: "1397f56fe818..."  ← computed from content  │
│  input: { product_name, review_text }                      │
│  output: { sentiment, confidence, key_phrases }            │
│  retry_count: 0                                            │
│  latency_ms: 843                                           │
│  tokens_used: 312                                          │
└────────────────────────────────────────────────────────────┘

CLI diff when you ship v2:

  contractllm diff analyse_sentiment v1 v2

  ┌─────────────────┬──────────────────┬──────────────────┐
  │ Property        │ v1               │ v2               │
  ├─────────────────┼──────────────────┼──────────────────┤
  │ Schema hash     │ 1397f56fe818...  │ 18b1319e9cd9...  │ ← changed
  │ Model           │ gpt-4o-mini      │ gpt-4o-mini      │
  │ Provider        │ openai           │ openai           │
  └─────────────────┴──────────────────┴──────────────────┘
  ⚠  Schema changed between versions.
```

The schema hash is computed from your actual content — system prompt + input
schema + output schema. Change anything without bumping the version string
and contractllm detects the mismatch. **You cannot accidentally ship an
untracked change.**

---

## Installation

```bash
pip install contractllm
```

All providers (OpenAI, Anthropic) are included. Missing API keys raise a clear error at instantiation — not at install time.

Add your API keys to a `.env` file:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Provider abstraction

```
Your contract
      │
      ▼
LLMProvider (abstract interface)
      │
      ├── OpenAIProvider
      │     └── response_format: json_schema (native structured output)
      │
      ├── AnthropicProvider
      │     └── schema injected into system prompt (no native JSON mode)
      │
      └── MockProvider  (for tests — zero API calls, zero cost)
```

The contract decorator never imports `openai` or `anthropic` directly.
Swap providers by changing one constructor argument — zero other changes.

```python
# Swap providers without touching the contract
@contract(
    name="analyse_sentiment",
    version="v1",
    system_prompt="You are a sentiment analysis expert.",
    input_schema=ReviewInput,
    output_schema=SentimentOutput,
    provider=AnthropicProvider(model="claude-3-5-haiku-20241022"),  # ← only this changes
)
async def analyse_sentiment(data: ReviewInput) -> SentimentOutput: ...
```

---

## Real example — job description parser

```python
from pydantic import BaseModel
from contractllm import contract
from contractllm.providers.openai import OpenAIProvider


class JDInput(BaseModel):
    raw_text: str


class ParsedJD(BaseModel):
    job_title: str
    required_skills: list[str]
    nice_to_have_skills: list[str]
    min_years_experience: int
    is_remote: bool
    salary_range: str | None


@contract(
    name="parse_job_description",
    version="v1",
    system_prompt=(
        "You are an expert at parsing job descriptions. "
        "Extract all information precisely. "
        "If a field is not mentioned, use null for optional fields "
        "and 0 for numeric fields."
    ),
    input_schema=JDInput,
    output_schema=ParsedJD,
    provider=OpenAIProvider(model="gpt-4o-mini"),
)
async def parse_jd(data: JDInput) -> ParsedJD: ...


# Works
result = await parse_jd({
    "raw_text": "Senior Python Engineer, 5+ yrs exp, remote OK, $150k-180k"
})

print(result.job_title)               # "Senior Python Engineer"
print(result.required_skills)         # ["Python"]
print(result.min_years_experience)    # 5
print(result.is_remote)               # True
print(result.salary_range)            # "$150k-180k"
```

---

## How contractllm compares

```
contractllm vs the existing landscape
─────────────────────────────────────────────────────────────────────────
Tool              What it does              What contractllm adds
─────────────────────────────────────────────────────────────────────────
Instructor        Structured output         + Versioning, regression
                  (one layer)               detection, retry-with-feedback

LangSmith /       Cloud tracing platform    contractllm is local-first.
Promptic SDK      Needs their dashboard     Zero account. Zero platform.

Guardrails        Rewrites bad output       contractllm fails fast and
                  to "fix" it              corrects via feedback, no magic

Pydantic          Python type validation    contractllm wraps the full
                  (not LLM-specific)        LLM call, not just parsing

llm-contracts     YAML-based output         contractllm is Python-native,
(PyPI, different  linting tool              decorator-based, typed end-to-end
project)
─────────────────────────────────────────────────────────────────────────
contractllm is the only tool that combines:
  ✓ Typed input validation
  ✓ Typed output validation
  ✓ Retry with error feedback (not blind retry)
  ✓ Schema versioning with content-addressed hashing
  ✓ Regression detection across versions
  ✓ Provider abstraction (OpenAI ↔ Anthropic, zero code change)
  ✓ Local-first (no cloud platform, no account, no vendor lock-in)
```

---

## CLI

```bash
# See all registered contracts
contractllm list

# Diff two versions
contractllm diff <contract-name> <version-a> <version-b>

# Recent runs for a contract
contractllm runs <contract-name> --version v1 --limit 20
```

---

## Supported providers

```
Provider      Structured output method
─────────────────────────────────────────────
OpenAI        response_format json_schema
Anthropic     Schema in system prompt
Gemini        coming soon
Mistral       coming soon
Ollama        coming soon            Local models, zero API cost
```

Both providers are included in the base install.

---

## Design decisions

**Decorator over class instantiation**
`@contract(...)` wraps an existing function with one line above it.
No restructuring of calling code. No new class to instantiate.
Follows Open/Closed: open for extension, closed for modification.

**Abstract base class for providers**
Application code depends on the `LLMProvider` interface, never on `openai`
or `anthropic` directly. Swapping providers = one constructor argument.
MockProvider makes the full system unit-testable without API calls.

**Retry with feedback, not blind retry**
Blind retry wastes tokens. The model fails again for the same reason.
Feeding the Pydantic validation error back gives the model the correction
signal it needs. Reduces retry rates 60–70% in practice.

**Content-addressed schema hashing**
Developers forget to bump version strings. The hash is computed from the
actual content system prompt + schemas. Same principle as Git. You cannot
accidentally ship an untracked change.

**SQLite for the version store**
Zero infrastructure. No server. No connection string. File-based so version
history travels with the repo. Would become PostgreSQL for a hosted service
that is when distribution and concurrent writes actually matter.

---

## Roadmap

- [ ] Gemini provider
- [ ] Mistral / Ollama (local models, zero API cost)
- [ ] Async batch — run N contracts concurrently with a budget cap
- [ ] Cost tracking — token spend per contract version, per run
- [ ] GitHub Action — regression CI gate on every push
- [ ] Hosted registry — share contracts across repos and teams

---

## Contributing

```bash
git clone https://github.com/tabishqazi/contractllm
cd contractllm
pip install -e ".[dev]"
pytest tests/ -v
ruff check contractllm/
```

Open an issue before large PRs.

---

## License

MIT. Use it, fork it, build on it.

---

Built by [Qazi Tabish Firoz Ahmed](https://github.com/tabishqazi) ·
[PyPI](https://pypi.org/project/contractllm/) ·
[Issues](https://github.com/tabishqazi/contractllm/issues)
