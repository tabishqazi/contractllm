"""
Same contract running on two different providers.
Demonstrates how the adapter pattern enables provider swapping.
Run: python examples/multi_provider.py
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
from contractllm.providers.anthropic import AnthropicProvider


class SummariseInput(BaseModel):
    article_text: str = Field(description="The article to summarise")
    max_sentences: int = Field(default=3, description="Maximum sentences in summary")


class SummariseOutput(BaseModel):
    headline: str
    summary: str
    bullet_points: list[str]


async def run_contract(provider, name: str):
    """Run the contract with a given provider and print results."""
    print(f"\n{'='*50}")
    print(f"Provider: {name}")
    print('='*50)

    @contract(
        name="summarise_article",
        version="v1",
        system_prompt="You are a precise news summarisation assistant.",
        input_schema=SummariseInput,
        output_schema=SummariseOutput,
        provider=provider,
    )
    async def summarise(data: SummariseInput) -> SummariseOutput:
        ...

    result = await summarise({
        "article_text": (
            "Scientists at MIT have developed a new type of solar cell "
            "that is 40% more efficient than existing technology. "
            "The breakthrough uses a novel material that captures light "
            "at different wavelengths simultaneously."
        ),
        "max_sentences": 2,
    })

    print(f"Headline: {result.headline}")
    print(f"Summary: {result.summary}")
    print(f"Bullet points: {result.bullet_points}")
    return result


async def main():
    # Run with OpenAI
    openai_provider = OpenAIProvider(model="gpt-4o-mini")
    await run_contract(openai_provider, "OpenAI (gpt-4o-mini)")

    # Run with Anthropic
    anthropic_provider = AnthropicProvider(model="claude-3-5-haiku-20241022")
    await run_contract(anthropic_provider, "Anthropic (claude-3-5-haiku)")


if __name__ == "__main__":
    asyncio.run(main())
