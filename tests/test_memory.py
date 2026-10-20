"""Integration tests: need a local Neo4j (skipped otherwise). SQLite uses a temp file."""
import asyncio
from datetime import date

import pytest

from app.config import get_settings
from app.db import neo4j as neo, sqlite as sq
from app.llm.types import Message, ToolCall, ToolResult
from app.models.memory import FetchArgs, MemoryOp
from app.models.user import ProfileUpdate, UserCreate
from app.repositories.brain_repo import BrainRepository
from app.repositories.session_repo import SessionRepository
from app.repositories.user_repo import UserExistsError, UserRepository
from app.services.context import build_system_prompt


def run(coro):
    return asyncio.run(coro)


async def _with_user(fn):
    await neo.init_neo4j()
    users, brain = UserRepository(neo.get_driver()), BrainRepository(neo.get_driver())
    user = await users.create(UserCreate(
        name="Test", date_of_birth=date(1995, 8, 15), birth_place="Patna", preferred_language="hindi"
    ))
    try:
        return await fn(user, brain)
    finally:
        await neo.get_driver().execute_query("MATCH (u:User {id: $id})-[*0..1]->(n) "
                                             "WHERE n:User OR n:Goal OR n:Memory DETACH DELETE n", id=user.id)
        await neo.close_neo4j()


@pytest.fixture(autouse=True, scope="module")
def require_neo4j():
    async def ping():
        await neo.init_neo4j()
        await neo.close_neo4j()

    try:
        run(ping())
    except Exception:
        pytest.skip("Neo4j not available")


def test_create_update_fetch_delete():
    async def body(user, brain):
        snap = await brain.snapshot(user.id)
        written = await brain.apply(user.id, [
            MemoryOp(op="create", type="goal", name="Switch to PM", summary="Move into PM", area="career",
                     target_year=2027),
            MemoryOp(op="create", type="interest", name="Entrepreneurship"),
            MemoryOp(op="create", type="preference", name="Short answers"),
        ], snap)
        assert [w["op"] for w in written] == ["create"] * 3
        goal_id = written[0]["ref"]

        # Prompt index shows titles only
        idx = await brain.prompt_index(user.id, 10)
        assert idx["goals"] == [{"id": goal_id, "name": "Switch to PM"}]
        assert idx["interests"] == ["Entrepreneurship"] and idx["preferences"] == ["Short answers"]
        system, refs, labels = build_system_prompt(user, idx)
        assert "Goal: Switch to PM" in system and "Move into PM" not in system
        assert labels[goal_id] == "goal:Switch to PM"

        # Duplicate create becomes update; status change hides it from active fetch
        snap = await brain.snapshot(user.id)
        written = await brain.apply(user.id, [
            MemoryOp(op="create", type="goal", name="switch to pm", status="achieved"),
        ], snap)
        assert written[0] == {"op": "update", "ref": goal_id, "label": "goal:switch to pm"}
        res, _, _ = await brain.fetch(user.id, FetchArgs(types=["goal"]))
        assert res["goal"] == []
        res, refs, _ = await brain.fetch(user.id, FetchArgs(types=["goal"], include_inactive=True, area="career"))
        assert res["goal"][0]["status"] == "achieved" and res["goal"][0]["target_year"] == 2027
        assert refs == [goal_id]

        # Delete goal and unlink interest
        snap = await brain.snapshot(user.id)
        written = await brain.apply(user.id, [
            MemoryOp(op="delete", type="goal", id=goal_id),
            MemoryOp(op="delete", type="interest", name="entrepreneurship"),
        ], snap)
        assert len(written) == 2
        snap = await brain.snapshot(user.id)
        assert {s["type"] for s in snap} == {"preference"}

    run(_with_user(body))


def test_rejects_bad_ops():
    async def body(user, brain):
        snap = await brain.snapshot(user.id)
        written = await brain.apply(user.id, [
            MemoryOp(op="update", type="goal", id="goal_doesnotexist", summary="x"),  # invented id
            MemoryOp(op="create", type="memory", id="x"),  # no name
            MemoryOp(op="update", type="interest", name="Chess"),  # update on shared type
            MemoryOp(op="create", type="life_area", name="astrophysics"),  # not a life area
            MemoryOp(op="delete", type="interest", name="Chess"),  # not linked
        ], snap)
        assert written == []

        # Memory id used as goal id -> type mismatch rejected
        w = await brain.apply(user.id, [MemoryOp(op="create", type="memory", name="Moved to Pune")], snap)
        snap = await brain.snapshot(user.id)
        bad = await brain.apply(user.id, [MemoryOp(op="delete", type="goal", id=w[0]["ref"])], snap)
        assert bad == []

    run(_with_user(body))


def test_profile_update_replaces_links_and_sign():
    async def body(user, brain):
        repo = UserRepository(neo.get_driver())
        u = await repo.update_profile(user.id, ProfileUpdate(
            preferred_language="english", birth_place="Delhi", date_of_birth=date(1990, 1, 25)
        ))
        assert (u.preferred_language, u.birth_place, u.sun_sign) == ("English", "Delhi", "Aquarius")
        assert u.name == "Test"  # untouched fields stay
        # Exactly one link of each kind remains
        recs, _, _ = await neo.get_driver().execute_query(
            "MATCH (:User {id: $id})-[r:PREFERS|BORN_IN|HAS_ZODIAC]->() RETURN type(r) AS t", id=user.id)
        assert sorted(r["t"] for r in recs) == ["BORN_IN", "HAS_ZODIAC", "PREFERS"]
        assert await repo.update_profile("no-such-user", ProfileUpdate(name="x")) is None

    run(_with_user(body))


def test_bare_user_and_goal_timeframe():
    async def body(user, brain):
        repo = UserRepository(neo.get_driver())
        uid = f"bare-{user.id[:8]}"
        try:
            bare = await repo.ensure(uid)
            assert (bare.name, bare.date_of_birth, bare.sun_sign, bare.preferred_language) == (None, None, None, "English")
            assert (await repo.ensure(uid)).id == uid  # idempotent
            with pytest.raises(UserExistsError):
                await repo.create(UserCreate(id=uid))

            w = await brain.apply(uid, [MemoryOp(op="create", type="goal", name="PM interview",
                                                 timeframe="next month", target_date="2026-11")], [])
            res, _, _ = await brain.fetch(uid, FetchArgs(types=["goal"]))
            assert (res["goal"][0]["timeframe"], res["goal"][0]["target_date"]) == ("next month", "2026-11")
            assert w[0]["op"] == "create"
        finally:
            await neo.get_driver().execute_query(
                "MATCH (u:User {id: $id}) OPTIONAL MATCH (u)-->(g:Goal) DETACH DELETE u, g", id=uid)

    run(_with_user(body))


def test_history_trims_to_user_start(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "sqlite_path", str(tmp_path / "t.db"))

    async def body():
        await sq.init_sqlite()
        repo = SessionRepository(sq.get_conn())
        sid = await repo.create("u1")
        await repo.add_messages(sid, "u1", [
            Message(role="user", text="hi"),
            Message(role="assistant", tool_calls=[ToolCall(id="t1", name="x")]),
            Message(role="tool", tool_results=[ToolResult(call_id="t1", name="x", content="{}")]),
            Message(role="assistant", text="hello"),
            Message(role="user", text="why?"),
        ])
        # Window of 3 would start at the tool row -> trimmed to the plain user message
        h = await repo.history(sid, 3)
        assert h == [Message(role="user", text="why?")]
        h = await repo.history(sid, 10)
        assert [m.role for m in h] == ["user", "assistant", "tool", "assistant", "user"]
        assert h[1].tool_calls[0].id == "t1"
        assert await repo.last_assistant_text(sid, 10**9) == "hello"
        # Client-chosen session ids: created on first use, owned by that user
        assert await repo.ensure("session-456", "u1") and await repo.ensure("session-456", "u1")
        assert not await repo.ensure("session-456", "someone-else")
        await sq.close_sqlite()

    run(body())
