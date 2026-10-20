import base64
import uuid

from google import genai
from google.genai import types

from app.llm.base import LLMProvider, T
from app.llm.types import Done, Message, TextDelta, ToolCall, ToolChoice, ToolSpec


def to_contents(messages: list[Message]) -> list[types.Content]:
    out = []
    for m in messages:
        if m.role == "tool":
            parts = [
                types.Part(function_response=types.FunctionResponse(
                    id=r.call_id, name=r.name, response={"result": r.content}
                ))
                for r in m.tool_results
            ]
            out.append(types.Content(role="user", parts=parts))
        elif m.role == "assistant":
            parts = [types.Part(text=m.text)] if m.text else []
            for c in m.tool_calls:
                sig = c.provider_data.get("thought_signature")
                parts.append(types.Part(
                    function_call=types.FunctionCall(id=c.id, name=c.name, args=c.args),
                    # Gemini requires its thought signature echoed back with the call
                    thought_signature=base64.b64decode(sig) if sig else None,
                ))
            out.append(types.Content(role="model", parts=parts))
        else:
            out.append(types.Content(role="user", parts=[types.Part(text=m.text)]))
    return out


def to_tools(tools: list[ToolSpec]) -> list[types.Tool]:
    decls = [
        types.FunctionDeclaration(name=t.name, description=t.description, parameters_json_schema=t.parameters)
        for t in tools
    ]
    return [types.Tool(function_declarations=decls)]


def _tool_call(part: types.Part) -> ToolCall:
    fc = part.function_call
    data = {"thought_signature": base64.b64encode(part.thought_signature).decode()} if part.thought_signature else {}
    # Older Gemini models don't return call ids
    return ToolCall(id=fc.id or f"call_{uuid.uuid4().hex[:8]}", name=fc.name, args=fc.args or {}, provider_data=data)


class GeminiProvider(LLMProvider):
    def __init__(self, model: str, api_key: str, max_retries: int = 3, base_url: str | None = None):
        super().__init__(model, api_key, max_retries)
        # SDK default is no retries; this retries 408/429/5xx and network errors with backoff
        retry = types.HttpRetryOptions(attempts=max_retries + 1, initial_delay=1.0, max_delay=10.0)
        self.client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(retry_options=retry, base_url=base_url)
        )

    async def stream(self, messages, system, tools, tool_choice: ToolChoice = "auto", max_tokens=1024):
        config = types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens)
        if tools:
            config.tools = to_tools(tools)
            config.tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=tool_choice.upper())
            )

        text, calls = "", []
        async for chunk in await self.client.aio.models.generate_content_stream(
            model=self.model, contents=to_contents(messages), config=config
        ):
            if not chunk.candidates or not chunk.candidates[0].content:
                continue
            for part in chunk.candidates[0].content.parts or []:
                if part.function_call:
                    calls.append(_tool_call(part))
                elif part.text and not part.thought:
                    text += part.text
                    yield TextDelta(part.text)
        yield Done(Message(role="assistant", text=text, tool_calls=calls))

    async def structured(self, system: str, prompt: str, schema: type[T], max_tokens=1024) -> T:
        resp = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                response_json_schema=schema.model_json_schema(),
            ),
        )
        return schema.model_validate_json(resp.text)
