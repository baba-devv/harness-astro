import json
import logging
from datetime import date

from app.llm.types import ToolSpec
from app.models.memory import FetchArgs
from app.models.user import User
from app.repositories.brain_repo import BrainRepository, shared_ref

log = logging.getLogger(__name__)

FETCH_TOOL = ToolSpec(
    name="get_user_memories",
    description=(
        "Fetch details from the user's long-term memory: goals, important memories, interests, "
        "preferences, life areas. Use when the question needs more than the titles in the memory index, "
        "or needs past memories. Not needed for small talk."
    ),
    parameters=FetchArgs.model_json_schema(),
)

_RULES = """You are a warm, personal assistant with light astrology knowledge.
Reply in the user's preferred language. Keep answers concise unless asked for depth.
Use the conversation history for follow-ups. Use the memory index to know what you know about
the user, and call get_user_memories when you need the details. Never invent facts about the user.
You cannot save memories yourself; what the user shares is remembered automatically.
If a profile field is unknown and you need it (e.g. date of birth for astrology), ask for it briefly;
otherwise answer without it.
Astrology: only the sun sign below is computed. Never state moon, rising, planet positions or houses
for the user; if they matter, say a full birth chart is needed."""

_DEGRADED = """
NOTE: long-term memory is temporarily unavailable. Use only this conversation; don't claim to
remember or not remember anything else about the user."""

EMPTY_INDEX = {"preferences": [], "interests": [], "life_areas": [], "goals": [], "memory_count": 0}


def _or_unknown(v) -> str:
    return str(v) if v is not None else "unknown"


def build_system_prompt(user: User, index: dict, degraded: bool = False) -> tuple[str, list[str], dict[str, str]]:
    """Returns (system prompt, injected refs, ref->label)."""
    tob = user.time_of_birth.strftime("%H:%M") if user.time_of_birth else None
    labels = {
        "user_profile": "user_profile",
        shared_ref("language", user.preferred_language): f"language:{user.preferred_language}",
    }

    lines = [
        _RULES + (_DEGRADED if degraded else ""),
        f"\nToday's date: {date.today().isoformat()}",
        "\n## User profile",
        f"- Name: {_or_unknown(user.name)}",
        f"- Date of birth: {_or_unknown(user.date_of_birth)}, time: {_or_unknown(tob)}",
        f"- Birth place: {_or_unknown(user.birth_place)}",
        f"- Sun sign: {_or_unknown(user.sun_sign)}",
        f"- Preferred language: {user.preferred_language}",
    ]
    if index["preferences"]:
        lines.append("\n## Preferences\n" + "\n".join(f"- {p}" for p in index["preferences"]))
        for p in index["preferences"]:
            labels[shared_ref("preference", p)] = f"preference:{p}"

    # Titles only; details come through the tool
    idx = [f"- Goal: {g['name']}" for g in index["goals"]]
    idx += [f"- Interest: {i}" for i in index["interests"]]
    idx += [f"- Life area in focus: {a}" for a in index["life_areas"]]
    if index["memory_count"]:
        idx.append(f"- {index['memory_count']} stored personal memories (fetch to see)")
    lines.append("\n## Memory index\n" + ("\n".join(idx) if idx else "- (empty)"))

    for g in index["goals"]:
        labels[g["id"]] = f"goal:{g['name']}"
    for i in index["interests"]:
        labels[shared_ref("interest", i)] = f"interest:{i}"
    for a in index["life_areas"]:
        labels[shared_ref("life_area", a)] = f"life_area:{a}"
    return "\n".join(lines), list(labels), labels


_UNAVAILABLE = "Long-term memory is temporarily unavailable. Answer from the conversation only."


def make_tool_handler(brain: BrainRepository | None, user_id: str, fetched: list[str], labels: dict[str, str]):
    """Tool executor bound to one user; records fetched refs for context_used. brain=None -> degraded."""

    async def handle(name: str, args: dict) -> str:
        if name != FETCH_TOOL.name:
            return f"Unknown tool {name}"
        if brain is None:
            return _UNAVAILABLE
        try:
            parsed = FetchArgs.model_validate(args)
        except ValueError as e:
            return f"Invalid arguments: {e}"
        try:
            result, refs, lbls = await brain.fetch(user_id, parsed)
        except Exception:
            log.exception("Memory fetch failed")
            return _UNAVAILABLE
        fetched.extend(r for r in refs if r not in fetched)
        labels.update(lbls)
        return json.dumps(result, ensure_ascii=False)

    return handle
