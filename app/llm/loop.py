from collections.abc import Awaitable, Callable

from app.llm.base import LLMProvider
from app.llm.types import Done, Message, TextDelta, ToolResult, ToolSpec

ToolHandler = Callable[[str, dict], Awaitable[str]]


async def stream_with_tools(
    llm: LLMProvider,
    messages: list[Message],
    system: str,
    tools: list[ToolSpec],
    on_tool: ToolHandler,
    new_rows: list[Message],
    max_rounds: int,
):
    """Provider-agnostic tool loop. Yields text; appends assistant/tool messages to `new_rows`."""
    msgs = list(messages)
    for round_ in range(max_rounds + 1):
        # Last round: no tools, so the loop always ends with a text answer
        choice = "none" if round_ == max_rounds else "auto"
        reply = None
        async for ev in llm.stream(msgs, system, tools, tool_choice=choice):
            if isinstance(ev, TextDelta):
                yield ev.text
            elif isinstance(ev, Done):
                reply = ev.message

        if reply.text or reply.tool_calls:
            new_rows.append(reply)
            msgs.append(reply)
        if not reply.tool_calls:
            return

        results = [ToolResult(call_id=c.id, name=c.name, content=await on_tool(c.name, c.args)) for c in reply.tool_calls]
        tool_msg = Message(role="tool", tool_results=results)
        new_rows.append(tool_msg)
        msgs.append(tool_msg)
