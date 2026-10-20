"""Provider-neutral types. Stored history uses these, so it's independent of any provider."""
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict  # JSON Schema


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict = Field(default_factory=dict)
    # Opaque data a provider needs echoed back (e.g. Gemini thought signatures)
    provider_data: dict = Field(default_factory=dict)


class ToolResult(BaseModel):
    call_id: str
    name: str
    content: str


class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)  # assistant only
    tool_results: list[ToolResult] = Field(default_factory=list)  # tool only


ToolChoice = Literal["auto", "none"]


@dataclass
class TextDelta:
    text: str


@dataclass
class Done:
    message: Message  # full assistant message, incl. any tool calls
