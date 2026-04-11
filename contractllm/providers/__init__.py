from contractllm.providers.base import LLMProvider, ProviderResponse

__all__ = ["LLMProvider", "ProviderResponse", "OpenAIProvider", "AnthropicProvider"]


def __getattr__(name: str):
    """Lazy-load provider classes so missing SDKs give clear install instructions."""
    if name == "OpenAIProvider":
        from contractllm.providers.openai import OpenAIProvider
        return OpenAIProvider
    if name == "AnthropicProvider":
        from contractllm.providers.anthropic import AnthropicProvider
        return AnthropicProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
