from neo4j import AsyncDriver, AsyncGraphDatabase

from app.config import get_settings

_driver: AsyncDriver | None = None

# Uniqueness constraints double as indexes for lookups
_CONSTRAINTS = [
    "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
    "CREATE CONSTRAINT place_name IF NOT EXISTS FOR (p:Place) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT language_name IF NOT EXISTS FOR (l:Language) REQUIRE l.name IS UNIQUE",
    "CREATE CONSTRAINT sign_name IF NOT EXISTS FOR (z:ZodiacSign) REQUIRE z.name IS UNIQUE",
    "CREATE CONSTRAINT goal_id IF NOT EXISTS FOR (g:Goal) REQUIRE g.id IS UNIQUE",
    "CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
    "CREATE CONSTRAINT interest_key IF NOT EXISTS FOR (i:Interest) REQUIRE i.key IS UNIQUE",
    "CREATE CONSTRAINT preference_key IF NOT EXISTS FOR (p:Preference) REQUIRE p.key IS UNIQUE",
    "CREATE CONSTRAINT area_name IF NOT EXISTS FOR (a:LifeArea) REQUIRE a.name IS UNIQUE",
]


async def init_neo4j() -> None:
    global _driver
    s = get_settings()
    # Unrecognized-label/rel warnings are noise on a fresh DB before any memory exists
    _driver = AsyncGraphDatabase.driver(
        s.neo4j_uri,
        auth=(s.neo4j_user, s.neo4j_password),
        notifications_disabled_classifications=["UNRECOGNIZED"],
    )
    await _driver.verify_connectivity()
    async with _driver.session() as session:
        for stmt in _CONSTRAINTS:
            await session.run(stmt)


async def close_neo4j() -> None:
    global _driver
    if _driver:
        await _driver.close()
        _driver = None


def get_driver() -> AsyncDriver:
    if _driver is None:
        raise RuntimeError("Neo4j driver not initialised")
    return _driver
