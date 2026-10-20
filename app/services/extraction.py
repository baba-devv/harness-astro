import json
import logging
from datetime import date

from app.db.neo4j import get_driver
from app.db.sqlite import get_conn
from app.llm import extraction_llm
from app.models.memory import ExtractionResult
from app.models.user import ProfileUpdate
from app.repositories.brain_repo import BrainRepository
from app.repositories.session_repo import SessionRepository
from app.repositories.user_repo import UserRepository

log = logging.getLogger(__name__)

_SYSTEM = """You maintain a user's long-term memory graph. Given the user's latest message, decide
whether it contains durable information worth remembering across future conversations.

Store: goals and plans, stated preferences (how they want answers, tone), interests,
important life events/memories, life areas they care about.
Profile: if the user states their own name, date/time of birth, birth place or wants a different
reply language, set those fields in "profile" (dates YYYY-MM-DD, times HH:MM). Only include fields
that are new or different from the current profile; otherwise leave profile null.
Do NOT store: small talk, questions, one-off requests, the assistant's statements, temporary moods.
Most messages need no changes: return an empty ops list.

Rules:
- Use "update" with the existing id when the message refines or changes an existing goal/memory.
- Use "delete" when the user asks to forget something or says it's no longer true
  (for a goal that was achieved or dropped, prefer update with status instead).
- Only use ids from the existing memory list. Never invent ids.
- Shared types (interest, preference, life_area) are addressed by name; there is no update,
  use delete + create.
- Goals: keep the user's wording in timeframe ("next month") and resolve it with today's date
  into target_date (YYYY-MM) and/or target_year.
- Keep names short (a few words) and summaries to one sentence, in English."""


def _changed_fields(user, update) -> list[str]:
    return [k for k, v in update.model_dump(exclude_none=True).items() if getattr(user, k) != v]


async def extract_and_store(user_id: str, session_id: str, message_id: int, message: str) -> None:
    """Runs in the background after each reply. Never raises; failures are logged."""
    try:
        brain, users = BrainRepository(get_driver()), UserRepository(get_driver())
        sessions = SessionRepository(get_conn())
        user = await users.get(user_id)
        snapshot = await brain.snapshot(user_id)
        # Previous assistant turn helps resolve replies like "yes, that one"
        prev = await sessions.last_assistant_text(session_id, message_id)

        prompt = (
            f"Today's date: {date.today().isoformat()}\n\n"
            f"Current profile:\n{user.model_dump_json(exclude={'id'})}\n\n"
            f"Existing memory:\n{json.dumps(snapshot, ensure_ascii=False, indent=1)}\n\n"
            f"Previous assistant message:\n{prev or '(none)'}\n\n"
            f"User's latest message:\n{message}"
        )
        result = await extraction_llm().structured(_SYSTEM, prompt, ExtractionResult)
        written = await brain.apply(user_id, result.ops, snapshot) if result.ops else []
        if result.profile:
            changed = _changed_fields(user, result.profile)
            if changed:
                await users.update_profile(user_id, ProfileUpdate(**result.profile.model_dump(include=set(changed))))
                written.append({"op": "update", "ref": "user_profile", "label": "profile:" + ",".join(changed)})
        await sessions.set_memory_writes(message_id, written)
        if written:
            log.info("Memory writes for message %s: %s", message_id, written)
    except Exception:
        log.exception("Extraction failed for message %s", message_id)
