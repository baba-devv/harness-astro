"""LLM layer tests: no network. A fake provider exercises the tool loop; converters are checked per provider."""
import asyncio
import base64

import pytest
from pydantic import BaseModel

from app.llm import PROVIDERS, Done, LLMProvider, Message, TextDelta, ToolCall, ToolResult, ToolSpec, get_provider
from app.llm.loop import stream_with_tools
from app.llm.providers import anthropic as anth, gemini as gem

TOOL = ToolSpec(name="lookup", description="d", parameters={"type": "object", "properties": {}})


class FakeProvider(LLMProvider):
    """Replays scripted assistant messages and records what it was sent."""

    def __init__(self, script: list[Message]):
        super().__init__("fake", "", 0)
        self.script, self.calls = list(script), []

    async def stream(self, messages, system, tools, tool_choice="auto", max_tokens=1024):
        self.calls.append((list(messages), tool_choice))
        reply = self.script.pop(0)
        if reply.text:
            yield TextDelta(reply.text)
        yield Done(reply)

    async def structured(self, system, prompt, schema, max_tokens=1024):
        raise NotImplementedError


def _run_loop(provider, max_rounds=3):
    async def go():
        rows, seen = [], []

        async def on_tool(name, args):
            seen.append((name, args))
            return "result"

        text = "".join([t async for t in stream_with_tools(
            provider, [Message(role="user", text="q")], "sys", [TOOL], on_tool, rows, max_rounds
        )])
        return text, rows, seen

    return asyncio.run(go())


def test_loop_runs_tool_then_answers():
    p = FakeProvider([
        Message(role="assistant", text="Checking. ", tool_calls=[ToolCall(id="c1", name="lookup", args={"a": 1})]),
        Message(role="assistant", text="Done."),
    ])
    text, rows, seen = _run_loop(p)
    assert text == "Checking. Done."
    assert seen == [("lookup", {"a": 1})]
    assert [r.role for r in rows] == ["assistant", "tool", "assistant"]
    assert rows[1].tool_results == [ToolResult(call_id="c1", name="lookup", content="result")]
    # Second call sees the tool exchange
    assert [m.role for m in p.calls[1][0]] == ["user", "assistant", "tool"]


def test_loop_forces_text_on_last_round():
    call = Message(role="assistant", tool_calls=[ToolCall(id="c", name="lookup")])
    p = FakeProvider([call, Message(role="assistant", text="final")])
    text, _, _ = _run_loop(p, max_rounds=1)
    assert text == "final"
    assert [c[1] for c in p.calls] == ["auto", "none"]


def test_registry():
    assert {"anthropic", "gemini"} <= set(PROVIDERS)
    with pytest.raises(ValueError):
        get_provider("nope", "m")
    assert get_provider("anthropic", "claude-x").model == "claude-x"


HISTORY = [
    Message(role="user", text="hi"),
    Message(role="assistant", text="let me check", tool_calls=[
        ToolCall(id="c1", name="lookup", args={"x": 1},
                 provider_data={"thought_signature": base64.b64encode(b"sig").decode()}),
    ]),
    Message(role="tool", tool_results=[ToolResult(call_id="c1", name="lookup", content="ok")]),
    Message(role="assistant", text="answer"),
]


def test_anthropic_conversion():
    out = anth.to_api_messages(HISTORY)
    assert out[0] == {"role": "user", "content": "hi"}
    assert out[1]["content"][1] == {"type": "tool_use", "id": "c1", "name": "lookup", "input": {"x": 1}}
    assert out[2] == {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "ok"}]}
    assert anth.to_api_tools([TOOL])[0]["input_schema"] == TOOL.parameters


def test_gemini_conversion():
    out = gem.to_contents(HISTORY)
    assert [c.role for c in out] == ["user", "model", "user", "model"]
    call_part = out[1].parts[1]
    assert call_part.function_call.name == "lookup" and call_part.function_call.args == {"x": 1}
    assert call_part.thought_signature == b"sig"
    fr = out[2].parts[0].function_response
    assert (fr.id, fr.name, fr.response) == ("c1", "lookup", {"result": "ok"})
    assert gem.to_tools([TOOL])[0].function_declarations[0].name == "lookup"


def test_gemini_tool_call_parse_keeps_signature():
    from google.genai import types

    part = types.Part(function_call=types.FunctionCall(name="lookup", args=None), thought_signature=b"s")
    c = gem._tool_call(part)
    assert c.args == {} and c.id.startswith("call_")
    assert base64.b64decode(c.provider_data["thought_signature"]) == b"s"


def test_schemas_are_json_serialisable():
    # Both providers receive Pydantic JSON schemas; make sure ours build cleanly
    from app.models.memory import ExtractionResult
    from app.services.context import FETCH_TOOL

    class Out(BaseModel):
        a: int

    for schema in (ExtractionResult.model_json_schema(), FETCH_TOOL.parameters, Out.model_json_schema()):
        assert schema["type"] == "object"
