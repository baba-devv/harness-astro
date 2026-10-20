import logging
import uuid
from typing import get_args

from neo4j import AsyncDriver

from app.models.memory import PER_USER_TYPES, FetchArgs, LifeArea, MemoryOp

log = logging.getLogger(__name__)

# type -> (label, relationship from User). Labels/rels come only from these maps,
# so the Cypher below stays fixed; all user/LLM values go in as parameters.
NODES = {"goal": ("Goal", "HAS_GOAL"), "memory": ("Memory", "HAS_MEMORY")}
# type -> (label, relationship, key property on the shared node)
SHARED = {
    "interest": ("Interest", "INTERESTED_IN", "key"),
    "preference": ("Preference", "PREFERS", "key"),
    "life_area": ("LifeArea", "FOCUSES_ON", "name"),
    "language": ("Language", "PREFERS", "name"),  # read-only here; set via the user profile
}
_LABEL_TO_TYPE = {v[0]: k for k, v in (NODES | SHARED).items()}
_AREAS = set(get_args(LifeArea))

_FETCH_NODES = """
MATCH (:User {{id: $uid}})-[:{rel}]->(n:{label})
WHERE ($include_inactive OR coalesce(n.status, 'active') = 'active')
  AND ($area IS NULL OR EXISTS {{ (n)-[:IN_AREA]->(:LifeArea {{name: $area}}) }})
RETURN n {{.id, .name, .summary, .timeframe, .target_date, .target_year, .status, .created_at, .updated_at,
          area: [(n)-[:IN_AREA]->(a) | a.name][0]}} AS item
ORDER BY n.updated_at DESC LIMIT 20
"""
_FETCH_LINKS = """
MATCH (:User {{id: $uid}})-[r:{rel}]->(n:{label})
RETURN n.name AS name, r.source AS source, r.created_at AS since
"""
_CREATE_NODE = """
MATCH (u:User {{id: $uid}})
CREATE (u)-[:{rel}]->(n:{label} {{id: $id, name: $name, summary: $summary, source: 'conversation',
                                created_at: datetime(), updated_at: datetime()}})
SET n += $extra
FOREACH (_ IN CASE WHEN $area IS NULL THEN [] ELSE [1] END |
  MERGE (a:LifeArea {{name: $area}}) MERGE (n)-[:IN_AREA]->(a))
"""
_UPDATE_NODE = """
MATCH (:User {{id: $uid}})-[:{rel}]->(n:{label} {{id: $id}})
SET n += $fields, n.updated_at = datetime()
WITH n
OPTIONAL MATCH (n)-[old:IN_AREA]->() WHERE $area IS NOT NULL
DELETE old
WITH DISTINCT n
FOREACH (_ IN CASE WHEN $area IS NULL THEN [] ELSE [1] END |
  MERGE (a:LifeArea {{name: $area}}) MERGE (n)-[:IN_AREA]->(a))
"""
_DELETE_NODE = "MATCH (:User {{id: $uid}})-[:{rel}]->(n:{label} {{id: $id}}) DETACH DELETE n"
_LINK = """
MATCH (u:User {{id: $uid}})
MERGE (n:{label} {{{key}: $key}}) ON CREATE SET n.name = $name
MERGE (u)-[r:{rel}]->(n) ON CREATE SET r.source = 'conversation', r.created_at = datetime()
"""
_UNLINK = "MATCH (:User {{id: $uid}})-[r:{rel}]->(:{label} {{{key}: $key}}) DELETE r"
_GOAL_FIELDS = ("timeframe", "target_date", "target_year", "status")


def _plain(v):
    """Neo4j temporal values -> ISO strings so results are JSON-serialisable."""
    return v.iso_format() if hasattr(v, "iso_format") else v


def _key(type_: str, name: str) -> str:
    name = name.strip()
    if type_ == "language":
        return name.title()
    if type_ == "life_area":
        return name
    return name.lower()


def shared_ref(type_: str, name: str) -> str:
    return f"{type_}:{_key(type_, name)}"


class BrainRepository:
    def __init__(self, driver: AsyncDriver):
        self.driver = driver

    async def _q(self, query: str, **params) -> list[dict]:
        records, _, _ = await self.driver.execute_query(query, **params)
        return [r.data() for r in records]

    async def prompt_index(self, user_id: str, limit: int) -> dict:
        """Always-on context: preferences in full, plus a capped title index of goals/interests/areas."""
        links = await self._q(
            "MATCH (:User {id: $uid})-[:PREFERS|INTERESTED_IN|FOCUSES_ON]->(n) "
            "RETURN labels(n)[0] AS label, n.name AS name",
            uid=user_id,
        )
        goals = await self._q(
            "MATCH (:User {id: $uid})-[:HAS_GOAL]->(g:Goal {status: 'active'}) "
            "RETURN g.id AS id, g.name AS name ORDER BY g.updated_at DESC LIMIT $limit",
            uid=user_id, limit=limit,
        )
        mem = await self._q(
            "MATCH (:User {id: $uid})-[:HAS_MEMORY]->(m:Memory) RETURN count(m) AS n", uid=user_id
        )
        by_type: dict[str, list[str]] = {}
        for l in links:
            by_type.setdefault(_LABEL_TO_TYPE[l["label"]], []).append(l["name"])
        return {
            "preferences": by_type.get("preference", []),
            "interests": by_type.get("interest", [])[:limit],
            "life_areas": by_type.get("life_area", []),
            "goals": goals,
            "memory_count": mem[0]["n"] if mem else 0,
        }

    async def fetch(self, user_id: str, args: FetchArgs) -> tuple[dict, list[str], dict[str, str]]:
        """Tool backend. Returns (result, refs, ref->label)."""
        types = args.types or ["goal", "memory", "interest", "preference", "life_area"]
        result, refs, labels = {}, [], {}
        for t in types:
            if t in NODES:
                label, rel = NODES[t]
                rows = await self._q(
                    _FETCH_NODES.format(rel=rel, label=label),
                    uid=user_id, area=args.area, include_inactive=args.include_inactive,
                )
                items = [{k: _plain(v) for k, v in r["item"].items() if v is not None} for r in rows]
                for it in items:
                    refs.append(it["id"])
                    labels[it["id"]] = f"{t}:{it['name']}"
            else:
                label, rel, _ = SHARED[t]
                rows = await self._q(_FETCH_LINKS.format(rel=rel, label=label), uid=user_id)
                items = [{k: _plain(v) for k, v in r.items() if v is not None} for r in rows]
                for it in items:
                    ref = shared_ref(t, it["name"])
                    refs.append(ref)
                    labels[ref] = f"{t}:{it['name']}"
            result[t] = items
        return result, refs, labels

    async def snapshot(self, user_id: str) -> list[dict]:
        """Everything extraction needs to decide create vs update: all per-user nodes + shared links."""
        nodes = await self._q(
            "MATCH (:User {id: $uid})-[:HAS_GOAL|HAS_MEMORY]->(n) "
            "RETURN n.id AS id, labels(n)[0] AS label, n.name AS name, n.summary AS summary, "
            "n.status AS status, n.timeframe AS timeframe, n.target_date AS target_date, "
            "n.target_year AS target_year, "
            "[(n)-[:IN_AREA]->(a) | a.name][0] AS area",
            uid=user_id,
        )
        links = await self._q(
            "MATCH (:User {id: $uid})-[:PREFERS|INTERESTED_IN|FOCUSES_ON]->(n) "
            "RETURN labels(n)[0] AS label, n.name AS name",
            uid=user_id,
        )
        out = []
        for n in nodes:
            item = {"type": _LABEL_TO_TYPE[n.pop("label")], **n}
            out.append({k: v for k, v in item.items() if v is not None})
        # Language lives in the profile, which extraction receives separately
        out += [{"type": _LABEL_TO_TYPE[l["label"]], "name": l["name"]} for l in links if l["label"] != "Language"]
        return out

    async def apply(self, user_id: str, ops: list[MemoryOp], snapshot: list[dict]) -> list[dict]:
        """Validates each op against the snapshot and runs fixed Cypher. Returns what was written."""
        node_types = {s["id"]: s["type"] for s in snapshot if "id" in s}
        node_names = {(s["type"], s["name"].lower()): s["id"] for s in snapshot if "id" in s}
        links = {shared_ref(s["type"], s["name"]) for s in snapshot if "id" not in s}
        written = []
        for op in ops:
            try:
                rec = await self._apply_one(user_id, op, node_types, node_names, links)
                if rec:
                    written.append(rec)
            except ValueError as e:
                log.warning("Rejected memory op %s: %s", op.model_dump(exclude_none=True), e)
        return written

    async def _apply_one(self, uid, op: MemoryOp, node_types, node_names, links) -> dict | None:
        t = op.type
        if t in PER_USER_TYPES:
            label, rel = NODES[t]
            # A create that duplicates an existing title becomes an update
            if op.op == "create" and op.name and (t, op.name.strip().lower()) in node_names:
                op = op.model_copy(update={"op": "update", "id": node_names[(t, op.name.strip().lower())]})

            if op.op == "create":
                if not op.name:
                    raise ValueError("create needs a name")
                nid = f"{t}_{uuid.uuid4().hex[:8]}"
                extra = {f: getattr(op, f) for f in _GOAL_FIELDS} | {"status": op.status or "active"} if t == "goal" else {}
                await self._q(
                    _CREATE_NODE.format(rel=rel, label=label),
                    uid=uid, id=nid, name=op.name.strip(), summary=op.summary or "",
                    extra=extra, area=op.area,
                )
                node_types[nid] = t
                node_names[(t, op.name.strip().lower())] = nid
                return {"op": "create", "ref": nid, "label": f"{t}:{op.name.strip()}"}

            # update/delete: the id must exist, belong to this user and match the type
            if not op.id or node_types.get(op.id) != t:
                raise ValueError(f"unknown or mismatched id {op.id!r} for type {t}")
            if op.op == "update":
                fields = {"name": op.name, "summary": op.summary}
                if t == "goal":
                    fields |= {f: getattr(op, f) for f in _GOAL_FIELDS}
                fields = {k: v for k, v in fields.items() if v is not None}
                await self._q(
                    _UPDATE_NODE.format(rel=rel, label=label), uid=uid, id=op.id, fields=fields, area=op.area
                )
                return {"op": "update", "ref": op.id, "label": f"{t}:{op.name or op.id}"}
            await self._q(_DELETE_NODE.format(rel=rel, label=label), uid=uid, id=op.id)
            node_types.pop(op.id, None)
            return {"op": "delete", "ref": op.id, "label": f"{t}:{op.id}"}

        # Shared concept nodes: create = link, delete = unlink
        if not op.name:
            raise ValueError("shared types need a name")
        if op.op == "update":
            raise ValueError("update not supported for shared types; use delete + create")
        if t == "life_area" and op.name.strip() not in _AREAS:
            raise ValueError(f"unknown life area {op.name!r}")
        label, rel, key_prop = SHARED[t]
        key, ref = _key(t, op.name), shared_ref(t, op.name)
        # When the node is keyed by name, the stored name must be the normalised key
        name = key if key_prop == "name" else op.name.strip()

        if op.op == "create":
            if ref in links:
                return None  # already linked; nothing to do
            await self._q(
                _LINK.format(label=label, key=key_prop, rel=rel), uid=uid, key=key, name=name
            )
            links.add(ref)
            return {"op": "create", "ref": ref, "label": f"{t}:{name}"}

        if ref not in links:
            raise ValueError(f"{ref} is not linked to this user")
        await self._q(_UNLINK.format(rel=rel, label=label, key=key_prop), uid=uid, key=key)
        links.discard(ref)
        return {"op": "delete", "ref": ref, "label": f"{t}:{name}"}
