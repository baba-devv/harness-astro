"""End-to-end tests through the HTTP API with the configured LLM provider(s), Neo4j and a temp SQLite.

Run: uv run pytest -m live                    (all)
     uv run pytest -m live -k full_conversation (main flow only)

Assertions check structure (graph state, memory_writes, context_used) rather than exact wording,
since LLM output varies. Background extraction finishes before TestClient returns, so the graph
can be checked right after each call.
"""
import json
import sqlite3
import uuid
from datetime import date

import pytest
from fastapi.testclient import TestClient
from neo4j import GraphDatabase

from app.config import get_settings

pytestmark = pytest.mark.live

_KEYS = {"anthropic": "anthropic_api_key", "gemini": "gemini_api_key"}


def _skip_reason() -> str | None:
    s = get_settings()
    for provider in {s.chat_provider, s.extraction_provider}:
        if not getattr(s, _KEYS.get(provider, ""), ""):
            return f"no API key for provider {provider!r}"
    try:
        with GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password)) as d:
            d.verify_connectivity()
    except Exception:
        return "Neo4j not available"
    return None


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    if reason := _skip_reason():
        pytest.skip(reason)
    s = get_settings()
    original = s.sqlite_path
    s.sqlite_path = str(tmp_path_factory.mktemp("e2e") / "e2e.db")
    from app.main import app

    with TestClient(app) as c:
        yield c
    s.sqlite_path = original


@pytest.fixture
def user_id():
    """Fresh user id per test; its graph data is removed afterwards."""
    uid = f"e2e-{uuid.uuid4().hex[:10]}"
    yield uid
    s = get_settings()
    with GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password)) as d:
        d.execute_query(
            "MATCH (u:User {id: $id}) OPTIONAL MATCH (u)-[:HAS_GOAL|HAS_MEMORY]->(n) DETACH DELETE u, n", id=uid
        )


def chat(client, uid: str, message: str, session_id: str) -> dict:
    r = client.post("/chat", json={"user_id": uid, "session_id": session_id, "message": message})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["response"].strip(), "empty LLM response"
    assert body["degraded"] is False
    return body


def last_writes(uid: str) -> list[dict]:
    """memory_writes of the user's latest message (None would mean extraction never finished)."""
    with sqlite3.connect(get_settings().sqlite_path) as conn:
        row = conn.execute(
            "SELECT memory_writes FROM messages WHERE user_id = ? AND role = 'user' ORDER BY id DESC LIMIT 1", (uid,)
        ).fetchone()
    assert row and row[0] is not None, "extraction did not run or failed (see logs)"
    return json.loads(row[0])


def goals(uid: str) -> list[dict]:
    s = get_settings()
    with GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password)) as d:
        recs, _, _ = d.execute_query(
            "MATCH (:User {id: $id})-[:HAS_GOAL]->(g:Goal) RETURN g {.*} AS g", id=uid
        )
    return [r["g"] for r in recs]


def session() -> str:
    return f"s-{uuid.uuid4().hex[:10]}"


def has_goal_label(body: dict) -> bool:
    return any(label.startswith("goal:") for label in body["context_used"])


def test_full_conversation(client, user_id):
    """New user -> memory created from chat -> used in context -> follow-up -> recalled in a new session."""
    s1, next_year = session(), date.today().year + 1

    # New user shares profile details and a plan in chat
    chat(client, user_id, "My name is Rahul. I was born on 15 August 1995 in Delhi. "
                          "I'm planning to switch jobs next year.", s1)
    profile = client.get(f"/users/{user_id}").json()
    assert profile["name"] == "Rahul"
    assert profile["date_of_birth"] == "1995-08-15"
    assert "delhi" in (profile["birth_place"] or "").lower()
    assert profile["sun_sign"] == "Leo"
    assert any(g.get("target_year") == next_year for g in goals(user_id)), goals(user_id)
    assert last_writes(user_id)

    # The stored goal is part of the selected context
    body = chat(client, user_id, "What should I focus on for my career?", s1)
    assert has_goal_label(body), body["context_used"]

    # Follow-up relies on session history
    body = chat(client, user_id, "Why do you say that?", s1)
    assert last_writes(user_id) == [], "a follow-up question should not create memories"

    # New session: long-term memory is retrieved
    body = chat(client, user_id, "What do you remember about my career goals?", session())
    assert has_goal_label(body), body["context_used"]
    text = body["response"].lower()
    assert any(w in text for w in ("job", "switch", str(next_year))), body["response"]


def test_irrelevant_message_not_stored(client, user_id):
    """Small talk must not become long-term memory."""
    chat(client, user_id, "Haha thanks, have a nice day!", session())
    assert last_writes(user_id) == []
    assert goals(user_id) == []


def test_correction_updates_existing_goal(client, user_id):
    """A correction updates the existing goal instead of adding a duplicate."""
    s = session()
    chat(client, user_id, "I'm planning to switch jobs next year.", s)
    assert len(goals(user_id)) == 1
    chat(client, user_id, "Actually, correction: I've pushed the job switch to 2028.", s)
    gs = goals(user_id)
    assert len(gs) == 1, gs
    assert gs[0]["target_year"] == 2028
    assert last_writes(user_id)[0]["op"] == "update"


def test_missing_profile_information(client, user_id):
    """No birth details: no crash, and the assistant asks instead of inventing a sign."""
    body = chat(client, user_id, "What does my zodiac sign say about my week ahead?", session())
    assert "birth" in body["response"].lower(), body["response"]
    assert client.get(f"/users/{user_id}").json()["sun_sign"] is None


def test_created_user_profile_is_used(client, user_id):
    """New user created via the API: profile is injected into context."""
    r = client.post("/users", json={"id": user_id, "name": "Asha", "date_of_birth": "1992-03-25",
                                    "birth_place": "Pune", "preferred_language": "English"})
    assert r.status_code == 201, r.text
    assert r.json()["sun_sign"] == "Aries"
    body = chat(client, user_id, "What's my sun sign?", session())
    assert "user_profile" in body["context_used"]
    assert "aries" in body["response"].lower()
