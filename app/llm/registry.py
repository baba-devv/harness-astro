from collections.abc import Callable
from functools import lru_cache

from app.config import get_settings
from app.llm.base import LLMProvider


# Imports are lazy so an unused provider's SDK never has to load
def _anthropic(model: str) -> LLMProvider:
    from app.llm.providers.anthropic import AnthropicProvider
    s = get_settings()
    return AnthropicProvider(model, s.anthropic_api_key, s.llm_max_retries)


def _gemini(model: str) -> LLMProvider:
    from app.llm.providers.gemini import GeminiProvider
    s = get_settings()
    return GeminiProvider(model, s.gemini_api_key, s.llm_max_retries)


PROVIDERS: dict[str, Callable[[str], LLMProvider]] = {
    "anthropic": _anthropic,
    "gemini": _gemini,
}


@lru_cache
def get_provider(name: str, model: str) -> LLMProvider:
    if name not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider {name!r}; available: {sorted(PROVIDERS)}")
    return PROVIDERS[name](model)


def chat_llm() -> LLMProvider:
    s = get_settings()
    return get_provider(s.chat_provider, s.chat_model)


def extraction_llm() -> LLMProvider:
    s = get_settings()
    return get_provider(s.extraction_provider, s.extraction_model)
