import json
import uuid

import aiosqlite

from app.llm.types import Message


class SessionRepository:
    def __init__(self, conn: aiosqlite.Connection):
        self.conn = conn

    async def create(self, user_id: str) -> str:
        sid = str(uuid.uuid4())
        await self.conn.execute("INSERT INTO sessions (id, user_id) VALUES (?, ?)", (sid, user_id))
        await self.conn.commit()
        return sid

    async def ensure(self, session_id: str, user_id: str) -> bool:
        """Creates the session if new (client-chosen ids allowed). False if owned by another user."""
        await self.conn.execute(
            "INSERT OR IGNORE INTO sessions (id, user_id) VALUES (?, ?)", (session_id, user_id)
        )
        await self.conn.commit()
        cur = await self.conn.execute("SELECT user_id FROM sessions WHERE id = ?", (session_id,))
        return (await cur.fetchone())["user_id"] == user_id

    async def add_messages(
        self, session_id: str, user_id: str, rows: list[Message], context_used: dict | None = None
    ) -> list[int]:
        """Appends messages in one transaction; context_used goes on the last row. Returns row ids."""
        cur = await self.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM messages WHERE session_id = ?", (session_id,)
        )
        seq = (await cur.fetchone())[0]
        ids = []
        for i, m in enumerate(rows):
            seq += 1
            ctx = context_used if context_used and i == len(rows) - 1 else None
            cur = await self.conn.execute(
                "INSERT INTO messages (session_id, user_id, seq, role, content, context_used) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    user_id,
                    seq,
                    m.role,
                    m.model_dump_json(exclude_defaults=True),
                    json.dumps(ctx) if ctx else None,
                ),
            )
            ids.append(cur.lastrowid)
        await self.conn.commit()
        return ids

    async def history(self, session_id: str, limit: int) -> list[Message]:
        """Last `limit` rows, trimmed so they start on a plain user message."""
        cur = await self.conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY seq DESC LIMIT ?",
            (session_id, limit),
        )
        rows = list(reversed(await cur.fetchall()))
        # Don't start mid tool exchange or on an assistant turn
        while rows and rows[0]["role"] != "user":
            rows.pop(0)
        return [Message.model_validate_json(r["content"]) for r in rows]

    async def last_assistant_text(self, session_id: str, before_id: int) -> str:
        cur = await self.conn.execute(
            "SELECT content FROM messages WHERE session_id = ? AND role = 'assistant' AND id < ? "
            "ORDER BY seq DESC LIMIT 1",
            (session_id, before_id),
        )
        row = await cur.fetchone()
        return Message.model_validate_json(row["content"]).text if row else ""

    async def set_memory_writes(self, message_id: int, refs: list[str]) -> None:
        await self.conn.execute(
            "UPDATE messages SET memory_writes = ? WHERE id = ?", (json.dumps(refs), message_id)
        )
        await self.conn.commit()
