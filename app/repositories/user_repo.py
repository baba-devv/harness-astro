import uuid

from neo4j import AsyncDriver, AsyncManagedTransaction
from neo4j.exceptions import ConstraintError

from app.models.user import ProfileUpdate, User, UserCreate
from app.services.astrology import sun_sign

# Scalar facts live on the User node; shared concepts (place, language, sign) are
# their own nodes so other users and future memories can link to them.
_CREATE = """
CREATE (u:User {id: $id, name: $name, date_of_birth: $dob, time_of_birth: $tob, created_at: datetime()})
"""

# Idempotent: creates a bare user (default language) only if the id is new
_ENSURE = """
MERGE (u:User {id: $id}) ON CREATE SET u.created_at = datetime()
WITH u WHERE NOT (u)-[:PREFERS]->(:Language)
MERGE (l:Language {name: $language})
CREATE (u)-[:PREFERS {source: 'default', created_at: datetime()}]->(l)
"""

# Replace the single link of a given type (birth place, language, sun sign)
_SET_LINK = """
MATCH (u:User {{id: $id}})
OPTIONAL MATCH (u)-[old:{rel}]->(:{label})
DELETE old
WITH DISTINCT u
MERGE (n:{label} {{name: $name}})
CREATE (u)-[r:{rel}]->(n)
SET r += $props, r.created_at = datetime()
"""
_LINKS = {
    "place": ("BORN_IN", "Place"),
    "language": ("PREFERS", "Language"),
    "sign": ("HAS_ZODIAC", "ZodiacSign"),
}

_GET = """
MATCH (u:User {id: $id})
OPTIONAL MATCH (u)-[:BORN_IN]->(p:Place)
OPTIONAL MATCH (u)-[:PREFERS]->(l:Language)
OPTIONAL MATCH (u)-[:HAS_ZODIAC {kind: 'sun'}]->(z:ZodiacSign)
RETURN u, p.name AS place, l.name AS language, z.name AS sign
"""


class UserExistsError(Exception):
    pass


def _norm_language(lang: str) -> str:
    return lang.strip().title()


async def _set_link(tx: AsyncManagedTransaction, user_id: str, kind: str, name: str, props: dict) -> None:
    rel, label = _LINKS[kind]
    await tx.run(_SET_LINK.format(rel=rel, label=label), id=user_id, name=name, props=props)


class UserRepository:
    def __init__(self, driver: AsyncDriver):
        self.driver = driver

    async def create(self, data: UserCreate) -> User:
        uid = data.id or str(uuid.uuid4())

        async def work(tx: AsyncManagedTransaction):
            await tx.run(_CREATE, id=uid, name=data.name, dob=data.date_of_birth, tob=data.time_of_birth)
            await _set_link(tx, uid, "language", _norm_language(data.preferred_language), {"source": "profile"})
            if data.birth_place:
                await _set_link(tx, uid, "place", data.birth_place.strip(), {})
            if data.date_of_birth:
                await _set_link(tx, uid, "sign", sun_sign(data.date_of_birth), {"kind": "sun"})

        async with self.driver.session() as s:
            try:
                await s.execute_write(work)
            except ConstraintError:
                raise UserExistsError(uid)
        return await self.get(uid)

    async def ensure(self, user_id: str) -> User:
        """Get the user, creating a bare profile on first contact."""
        await self.driver.execute_query(_ENSURE, id=user_id, language="English")
        return await self.get(user_id)

    async def update_profile(self, user_id: str, data: ProfileUpdate) -> User | None:
        props = {k: v for k, v in {"name": data.name, "date_of_birth": data.date_of_birth,
                                   "time_of_birth": data.time_of_birth}.items() if v is not None}

        async def work(tx: AsyncManagedTransaction) -> bool:
            res = await tx.run("MATCH (u:User {id: $id}) SET u += $props RETURN u.id", id=user_id, props=props)
            if not await res.single():
                return False
            if data.birth_place:
                await _set_link(tx, user_id, "place", data.birth_place.strip(), {})
            if data.preferred_language:
                await _set_link(tx, user_id, "language", _norm_language(data.preferred_language),
                                {"source": "conversation"})
            if data.date_of_birth:  # sign follows date of birth
                await _set_link(tx, user_id, "sign", sun_sign(data.date_of_birth), {"kind": "sun"})
            return True

        async with self.driver.session() as s:
            if not await s.execute_write(work):
                return None
        return await self.get(user_id)

    async def get(self, user_id: str) -> User | None:
        records, _, _ = await self.driver.execute_query(_GET, id=user_id)
        if not records:
            return None
        r = records[0]
        u = r["u"]
        # Neo4j returns its own temporal types; convert back to stdlib
        dob, tob = u.get("date_of_birth"), u.get("time_of_birth")
        return User(
            id=u["id"],
            name=u.get("name"),
            date_of_birth=dob.to_native() if dob else None,
            time_of_birth=tob.to_native() if tob else None,
            birth_place=r["place"],
            preferred_language=r["language"] or "English",
            sun_sign=r["sign"],
        )
