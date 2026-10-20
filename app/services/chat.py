import logging
import uuid
from dataclasses import dataclass, field

from neo4j.exceptions import DriverError, Neo4jError

from app.config import get_settings
from app.db.neo4j import get_driver
from app.db.sqlite import get_conn
from app.llm import Message, chat_llm
from app.llm.loop import ToolHandler, stream_with_tools
from app.models.user import User
from app.repositories.brain_repo import BrainRepository
from app.repositories.session_repo import SessionRepository
from app.repositories.user_repo import UserRepository
from app.services.context import EMPTY_INDEX, FETCH_TOOL, build_system_prompt, make_tool_handler

log = logging.getLogger(__name__)


class SessionOwnershipError(Exception):
    pass


@dataclass
class Turn:
    """A chat turn with the user message saved and the prompt built. `stream()` calls the LLM and saves the reply."""

    user_id: str
    session_id: str
    message_id: int
    system: str
    history: list[Message]
    on_tool: ToolHandler
    injected: list[str]
    labels: dict[str, str]
    degraded: bool
    fetched: list[str] = field(default_factory=list)
    new_rows: list[Message] = field(default_factory=list)

    async def stream(self):
        async for text in stream_with_tools(
            chat_llm(), self.history, self.system, [FETCH_TOOL], self.on_tool,
            self.new_rows, get_settings().max_tool_rounds,
        ):
            yield text
        if self.new_rows:
            await SessionRepository(get_conn()).add_messages(
                self.session_id, self.user_id, self.new_rows,
                {"injected": self.injected, "fetched": self.fetched},
            )

    def context_used(self) -> dict:
        """Readable labels; raw refs stay in SQLite."""
        return {
            "injected": [self.labels.get(r, r) for r in self.injected],
            "fetched": [self.labels.get(r, r) for r in self.fetched],
        }


async def start_turn(user_id: str, session_id: str | None, message: str) -> Turn:
    s = get_settings()
    sessions = SessionRepository(get_conn())
    session_id = session_id or str(uuid.uuid4())
    if not await sessions.ensure(session_id, user_id):
        raise SessionOwnershipError(session_id)

    # Save the user message first so it survives an LLM failure
    [msg_id] = await sessions.add_messages(session_id, user_id, [Message(role="user", text=message)])

    # Profile + memory index from Neo4j; fall back to history-only if the graph is down
    degraded = False
    try:
        brain = BrainRepository(get_driver())
        user = await UserRepository(get_driver()).ensure(user_id)
        index = await brain.prompt_index(user_id, s.index_limit)
    except (Neo4jError, DriverError, RuntimeError):
        log.exception("Shared Brain unavailable; answering from conversation only")
        degraded, brain, user, index = True, None, User(id=user_id), EMPTY_INDEX

    system, injected, labels = build_system_prompt(user, index, degraded)
    if degraded:
        injected = []  # nothing came from the brain
    history = await sessions.history(session_id, s.history_limit)

    fetched: list[str] = []
    return Turn(
        user_id=user_id, session_id=session_id, message_id=msg_id, system=system, history=history,
        on_tool=make_tool_handler(brain, user_id, fetched, labels), injected=injected, labels=labels,
        degraded=degraded, fetched=fetched,
    )
