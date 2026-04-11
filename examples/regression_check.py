"""
Regression check: compare two versions of the same contract.
Shows how schema hashes detect content drift.
Run: python examples/regression_check.py
"""
import asyncio
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv() -> bool:
        return False

load_dotenv()

from contractllm import contract
from contractllm.providers.openai import OpenAIProvider
from contractllm.store.version_store import VersionStore


class ExtractInput(BaseModel):
    job_description: str = Field(description="The job posting text")


class ExtractOutput(BaseModel):
    required_skills: list[str]
    preferred_skills: list[str]
    experience_level: str  # "junior", "mid", "senior", "lead"


provider = OpenAIProvider(model="gpt-4o-mini")
store = VersionStore()


# Version 1: Original prompt
@contract(
    name="extract_job_skills",
    version="v1",
    system_prompt="You are a job posting analyser. Extract skills from job descriptions.",
    input_schema=ExtractInput,
    output_schema=ExtractOutput,
    provider=provider,
    store=store,
)
async def extract_skills_v1(data: ExtractInput) -> ExtractOutput:
    ...


# Version 2: Improved prompt
@contract(
    name="extract_job_skills",
    version="v2",
    system_prompt=(
        "You are a precise job posting analyst. "
        "Extract required and preferred skills, and determine experience level. "
        "Be thorough and specific in skill names."
    ),
    input_schema=ExtractInput,
    output_schema=ExtractOutput,
    provider=provider,
    store=store,
)
async def extract_skills_v2(data: ExtractInput) -> ExtractOutput:
    ...


async def main():
    job_text = (
        "We are looking for a Senior Software Engineer to join our team. "
        "Required: 5+ years Python, experience with PostgreSQL and AWS. "
        "Preferred: Kubernetes, Go, React."
    )

    print("Running v1...")
    result_v1 = await extract_skills_v1({"job_description": job_text})
    print(f"V1 required_skills: {result_v1.required_skills}")

    print("\nRunning v2...")
    result_v2 = await extract_skills_v2({"job_description": job_text})
    print(f"V2 required_skills: {result_v2.required_skills}")

    # Compare versions in the store
    def_v1 = store.get_definition("extract_job_skills", "v1")
    def_v2 = store.get_definition("extract_job_skills", "v2")

    print(f"\n{'='*50}")
    print("Regression Report")
    print('='*50)
    print(f"V1 schema hash: {def_v1.schema_hash}")
    print(f"V2 schema hash: {def_v2.schema_hash}")
    print(f"Schema changed: {def_v1.schema_hash != def_v2.schema_hash}")

    # Show recent runs
    runs_v1 = store.get_runs("extract_job_skills", "v1", limit=3)
    runs_v2 = store.get_runs("extract_job_skills", "v2", limit=3)
    print(f"\nV1 total runs: {len(runs_v1)}")
    print(f"V2 total runs: {len(runs_v2)}")


if __name__ == "__main__":
    asyncio.run(main())
