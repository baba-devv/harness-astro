"""Retries: a local server fails with 503 first, then succeeds. No network."""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from anthropic import AsyncAnthropic
from pydantic import BaseModel

from app.llm import Message
from app.llm.providers.anthropic import AnthropicProvider
from app.llm.providers.gemini import GeminiProvider


class Out(BaseModel):
    a: int


GEMINI_BODY = {"candidates": [{"content": {"role": "model", "parts": [{"text": '{"a": 1}'}]}}]}
ANTHROPIC_BODY = {
    "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
    "content": [{"type": "tool_use", "id": "t1", "name": "output", "input": {"a": 1}}],
    "stop_reason": "tool_use", "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


@pytest.fixture
def flaky_server():
    """Returns (base_url, state). Fails the first state['fail'] requests with 503."""
    state = {"fail": 1, "hits": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            state["hits"] += 1
            if state["hits"] <= state["fail"]:
                return self._send(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}})
            if "streamGenerateContent" in self.path:
                body = f"data: {json.dumps(GEMINI_BODY)}\n\n".encode()
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            self._send(200, GEMINI_BODY if "generateContent" in self.path else ANTHROPIC_BODY)

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", state
    srv.shutdown()


def test_gemini_structured_retries(flaky_server):
    url, state = flaky_server
    p = GeminiProvider("m", "k", max_retries=2, base_url=url)
    assert asyncio.run(p.structured("s", "p", Out)) == Out(a=1)
    assert state["hits"] == 2


def test_gemini_stream_retries(flaky_server):
    url, state = flaky_server
    p = GeminiProvider("m", "k", max_retries=2, base_url=url)

    async def go():
        return [ev async for ev in p.stream([Message(role="user", text="hi")], "s", [])]

    events = asyncio.run(go())
    assert events[-1].message.text == '{"a": 1}'
    assert state["hits"] == 2


def test_gemini_gives_up_after_max_retries(flaky_server):
    url, state = flaky_server
    state["fail"] = 99
    p = GeminiProvider("m", "k", max_retries=1, base_url=url)
    with pytest.raises(Exception):
        asyncio.run(p.structured("s", "p", Out))
    assert state["hits"] == 2  # 1 try + 1 retry


def test_anthropic_retries(flaky_server):
    url, state = flaky_server
    p = AnthropicProvider("m", "k", max_retries=2)
    p.client = AsyncAnthropic(api_key="k", base_url=url, max_retries=p.max_retries)
    assert asyncio.run(p.structured("s", "p", Out)) == Out(a=1)
    assert state["hits"] == 2
