from pathlib import Path

import aiosqlite

from app.config import get_settings

_conn: aiosqlite.Connection | None = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One row per message in order. role: user | assistant | tool (tool results).
-- content is a provider-neutral app.llm.Message as JSON.
CREATE TABLE IF NOT EXISTS messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(id),
    user_id       TEXT NOT NULL,
    seq           INTEGER NOT NULL,
    role          TEXT NOT NULL,
    content       TEXT NOT NULL,
    context_used  TEXT,  -- final assistant row: {"injected": [...], "fetched": [...]}
    memory_writes TEXT,  -- user row: refs written by background extraction
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (session_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""


async def init_sqlite() -> None:
    global _conn
    path = Path(get_settings().sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _conn = await aiosqlite.connect(path)
    _conn.row_factory = aiosqlite.Row
    await _conn.execute("PRAGMA journal_mode=WAL")
    await _conn.executescript(_SCHEMA)
    await _conn.commit()


async def close_sqlite() -> None:
    global _conn
    if _conn:
        await _conn.close()
        _conn = None


def get_conn() -> aiosqlite.Connection:
    if _conn is None:
        raise RuntimeError("SQLite connection not initialised")
    return _conn
