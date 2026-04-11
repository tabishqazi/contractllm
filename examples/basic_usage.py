"""
Simplest possible example of llm-contracts.
Run: python examples/basic_usage.py
"""
import asyncio
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv() -> bool:
        return False

load_dotenv()

from contractllm import contract
from contractllm.providers.openai import OpenAIProvider


class ReviewInput(BaseModel):
    product_name: str
    review_text: str


class SentimentOutput(BaseModel):
    sentiment: str          # "positive", "negative", "neutral"
    confidence: float       # 0.0 to 1.0
    key_phrases: list[str]  # Top phrases that drove the sentiment


provider = OpenAIProvider(model="gpt-4o-mini")


@contract(
    name="analyse_sentiment",
    version="v1",
    system_prompt=(
        "You are a sentiment analysis expert. "
        "Analyse product reviews and extract structured insights."
    ),
    input_schema=ReviewInput,
    output_schema=SentimentOutput,
    provider=provider,
)
async def analyse_sentiment(data: ReviewInput) -> SentimentOutput:
    ...  # Body is replaced by the decorator


async def main():
    result = await analyse_sentiment({
        "product_name": "Wireless Headphones X1",
        "review_text": (
            "These headphones are absolutely fantastic. "
            "The sound quality is incredible and battery life lasts forever. "
            "Only minor complaint is they're a bit heavy."
        ),
    })

    print(f"Sentiment: {result.sentiment}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Key phrases: {', '.join(result.key_phrases)}")


if __name__ == "__main__":
    asyncio.run(main())
