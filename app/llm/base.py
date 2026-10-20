from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TypeVar

from pydantic import BaseModel

from app.llm.types import Done, Message, TextDelta, ToolChoice, ToolSpec

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Implement these two methods to add a provider, then register it in registry.py."""

    def __init__(self, model: str, api_key: str, max_retries: int = 3):
        # Providers should retry transient errors (429/5xx/network) up to max_retries times
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        system: str,
        tools: list[ToolSpec],
        tool_choice: ToolChoice = "auto",
        max_tokens: int = 1024,
    ) -> AsyncIterator[TextDelta | Done]:
        """Yield TextDelta as text arrives, then exactly one Done with the full assistant message."""

    @abstractmethod
    async def structured(self, system: str, prompt: str, schema: type[T], max_tokens: int = 1024) -> T:
        """Single call whose output is validated into `schema`."""
