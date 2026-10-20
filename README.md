# Harness

Astrology chat service that remembers users across conversations.

FastAPI for the API, Neo4j for long-term user memory, SQLite for per-session chat history, and a
small LLM layer that works with Anthropic or Gemini.

## Running it

Needs Python 3.12+, [uv](https://docs.astral.sh/uv/) and Neo4j 5+.

```bash
# Neo4j
brew install neo4j
neo4j-admin dbms set-initial-password harness-dev
neo4j start
# or: docker run -d -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/harness-dev neo4j:5

# App
uv sync
cp .env.example .env    # add your API key, pick provider/models
uv run uvicorn app.main:app --reload
```

API docs at http://localhost:8000/docs.

Main settings in `.env`:

```
CHAT_PROVIDER=anthropic            # or gemini
CHAT_MODEL=claude-sonnet-5-5
EXTRACTION_PROVIDER=anthropic      # model that decides what to remember
EXTRACTION_MODEL=claude-haiku-4-5-20251001
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
```

Other knobs (retries, history size, Neo4j/SQLite location) are in `app/config.py`.

## API

- `POST /chat`: send a message
- `POST /users`, `GET /users/{id}`, `PATCH /users/{id}`: profile
- `GET /health`

Creating a user up front is optional. An unknown `user_id` gets an empty profile on its first message,
and details like name or birth date shared in chat are picked up automatically. Session ids can be
chosen by the client; a new one starts a new session.

```bash
curl -X POST localhost:8000/chat -H 'content-type: application/json' -d '{
  "user_id": "user-123",
  "session_id": "session-456",
  "message": "What should I focus on for my career?"
}'
```

```json
{
  "response": "As a Leo, you do best when you own visible work. With the job switch planned for next year...",
  "user_id": "user-123",
  "session_id": "session-456",
  "context_used": ["user_profile", "language:English", "goal:Switch jobs"],
  "context_detail": {"injected": ["user_profile", "language:English", "goal:Switch jobs"], "fetched": []},
  "degraded": false
}
```

`injected` is what's always in the prompt, `fetched` is what the model pulled from memory on its own.
Add `"stream": true` to get Server-Sent Events instead.

## How a message flows

1. Save the user message to SQLite.
2. Load the profile and a short memory index (titles only) from Neo4j, plus the last 20 messages of
   the session.
3. Call the chat model. If it needs details (e.g. "what do you remember about my goals?") it calls the
   `get_user_memories` tool, which runs a fixed Cypher query.
4. Return the answer and save it, with any tool calls and the context used.
5. In the background, an extraction model looks at the user's message and existing memory and decides
   whether anything is worth keeping. Its output is a list of create/update/delete ops, which get
   validated and written to Neo4j.

## Memory

**Short-term**: the session's messages in SQLite. This is what handles follow-ups like "why do you say
that?".

**Long-term**: a graph per user in Neo4j.

```
(User)-[:HAS_GOAL]->(Goal)-[:IN_AREA]->(LifeArea)
(User)-[:HAS_MEMORY]->(Memory)-[:IN_AREA]->(LifeArea)
(User)-[:INTERESTED_IN]->(Interest)
(User)-[:PREFERS]->(Preference | Language)
(User)-[:FOCUSES_ON]->(LifeArea)
(User)-[:HAS_ZODIAC]->(ZodiacSign)
(User)-[:BORN_IN]->(Place)
```

Goals and memories belong to one user. Interests, languages, signs, places and life areas are shared
nodes that many users link to; anything user-specific (source, when) sits on the relationship.

For example, "I'm planning to switch jobs next year" becomes
`Goal {name: "Switch jobs", timeframe: "next year", target_year: 2027, status: "active"}`.

What gets stored: goals, preferences, interests, important events, profile facts. What doesn't: small
talk, questions, one-off requests. Corrections update the existing node rather than adding a new one,
and "forget X" deletes it. The extractor can only reference ids that exist for that user, and all
writes go through fixed Cypher, so the LLM never writes queries.

Why a graph: user facts are connected (goal → life area, many users → one interest) and the schema can
grow without migrations.

## Context selection

The whole graph never goes into the prompt. Every turn gets the profile, preferences and a capped list
of goal/interest titles. Details only come in when the model asks for them through the tool. If the
profile is missing something (e.g. birth date) the model is told it's unknown and asks when it needs
it. Only the sun sign is computed; the model is told not to make up other placements.

## When things fail

- LLM errors are retried with backoff. If the call still fails, `/chat` returns 503 and the user's
  message is kept, so a retry works.
- If Neo4j is down, chat still answers from the session history and returns `degraded: true`.
- Extraction failures are logged and don't affect the reply.
- Bad input gets a 422; using another user's session gets a 403.

## Tests

```bash
uv run pytest          # unit + integration, no LLM calls (Neo4j tests skip if it isn't running)
uv run pytest -m live  # end-to-end through the API with real LLM calls
```

The live tests use whatever provider is set in `.env` and skip if there's no key. They cover:

- a full conversation: new user shares details and a plan, asks for advice, asks a follow-up, then
  a new session recalls the goal
- small talk not being stored
- a correction updating the existing goal
- missing birth details (assistant asks instead of guessing)
- a user created through the API having their profile used

They check graph state and `context_used` rather than exact wording. A full run is about 20-25 LLM
calls; free-tier Gemini keys run out at around 20 a day per model.

## Evaluating memory quality

Not built, but the approach would be: a small labelled set of messages with the expected memory ops,
run through extraction to get precision/recall (including how often small talk wrongly gets stored
and whether corrections become updates). For retrieval, seed a user's memory and check whether the
relevant item shows up in `context_used`. For personalization, compare answers with and without memory
using a judge model. `memory_writes` and `context_used` are stored per message, so most of this can be
computed from existing data.

## Trade-offs and what I'd change for production

- Memory is written in the background, so it's available from the next turn, and lost if the process
  dies mid-extraction. In production this should go through a queue.
- SQLite is fine for one instance; multiple instances would need Postgres or Redis.
- History is a fixed 20-message window; a token budget with summarization would be better.
- No auth yet: user ids are trusted.
- Per-user memory is small, so a title index plus a fetch tool works well. Embedding search would make
  sense once memory gets large.
