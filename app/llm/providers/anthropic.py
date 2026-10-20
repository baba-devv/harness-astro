from anthropic import AsyncAnthropic

from app.llm.base import LLMProvider, T
from app.llm.types import Done, Message, TextDelta, ToolCall, ToolChoice, ToolSpec


def to_api_messages(messages: list[Message]) -> list[dict]:
    out = []
    for m in messages:
        if m.role == "tool":
            blocks = [
                {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content}
                for r in m.tool_results
            ]
            out.append({"role": "user", "content": blocks})
        elif m.role == "assistant":
            blocks = [{"type": "text", "text": m.text}] if m.text else []
            blocks += [
                {"type": "tool_use", "id": c.id, "name": c.name, "input": c.args} for c in m.tool_calls
            ]
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": "user", "content": m.text})
    return out


def to_api_tools(tools: list[ToolSpec]) -> list[dict]:
    return [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str, api_key: str, max_retries: int = 3):
        super().__init__(model, api_key, max_retries)
        # SDK retries 408/409/429/5xx and connection errors with backoff
        self.client = AsyncAnthropic(api_key=api_key, max_retries=max_retries)

    async def stream(self, messages, system, tools, tool_choice: ToolChoice = "auto", max_tokens=1024):
        kwargs = {"tools": to_api_tools(tools), "tool_choice": {"type": tool_choice}} if tools else {}
        async with self.client.messages.stream(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=to_api_messages(messages), **kwargs,
        ) as s:
            async for text in s.text_stream:
                yield TextDelta(text)
            final = await s.get_final_message()

        text = "".join(b.text for b in final.content if b.type == "text")
        calls = [ToolCall(id=b.id, name=b.name, args=b.input) for b in final.content if b.type == "tool_use"]
        yield Done(Message(role="assistant", text=text, tool_calls=calls))

    async def structured(self, system: str, prompt: str, schema: type[T], max_tokens=1024) -> T:
        # Forced tool call is Anthropic's most reliable structured-output path
        tool = {"name": "output", "description": "Return the result.", "input_schema": schema.model_json_schema()}
        resp = await self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
            tools=[tool], tool_choice={"type": "tool", "name": "output"},
        )
        return schema.model_validate(next(b.input for b in resp.content if b.type == "tool_use"))
