import json
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from app.models.user import ID_PATTERN
from app.services.chat import SessionOwnershipError, start_turn
from app.services.extraction import extract_and_store

log = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    user_id: str = Field(pattern=ID_PATTERN)  # unknown ids get a bare profile on first message
    session_id: str | None = Field(None, pattern=ID_PATTERN)  # omit to start a new session
    message: str = Field(min_length=1, max_length=4000)
    stream: bool = False  # true -> Server-Sent Events


class ChatResponse(BaseModel):
    response: str
    user_id: str
    session_id: str
    context_used: list[str]  # flat list of labels, e.g. ["user_profile", "goal:Switch to PM"]
    context_detail: dict  # {"injected": [...], "fetched": [...]}
    degraded: bool = False  # true when the Shared Brain was unavailable


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("", response_model=ChatResponse, responses={503: {"description": "LLM unavailable"}})
async def chat(req: ChatRequest, background: BackgroundTasks):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message is empty")
    try:
        turn = await start_turn(req.user_id, req.session_id, message)
    except SessionOwnershipError:
        raise HTTPException(status_code=403, detail="Session belongs to another user")

    # Memory extraction runs after the response is sent, in both modes
    extract_args = (extract_and_store, turn.user_id, turn.session_id, turn.message_id, message)

    if req.stream:
        async def events():
            yield _sse("session", {"session_id": turn.session_id, "degraded": turn.degraded})
            try:
                async for text in turn.stream():
                    yield _sse("delta", {"text": text})
            except Exception as e:
                log.exception("Chat stream failed")
                yield _sse("error", {"detail": f"LLM unavailable: {e}"})
                return
            yield _sse("context", turn.context_used())
            yield _sse("done", {})

        return StreamingResponse(events(), media_type="text/event-stream", background=BackgroundTask(*extract_args))

    background.add_task(*extract_args)
    try:
        text = "".join([t async for t in turn.stream()])
    except Exception:
        log.exception("Chat failed")
        # User message is saved; the client can simply retry
        return JSONResponse(status_code=503, content={
            "detail": "The assistant is temporarily unavailable. Please try again.",
            "user_id": turn.user_id, "session_id": turn.session_id,
        })
    ctx = turn.context_used()
    return ChatResponse(
        response=text, user_id=turn.user_id, session_id=turn.session_id,
        context_used=list(dict.fromkeys(ctx["injected"] + ctx["fetched"])),
        context_detail=ctx, degraded=turn.degraded,
    )
