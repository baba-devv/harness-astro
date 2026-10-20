from typing import Literal

from pydantic import BaseModel, Field

from app.models.user import ProfileUpdate

LifeArea = Literal[
    "career", "finance", "health", "relationships", "family",
    "education", "spirituality", "personal_growth",
]
MemoryType = Literal["goal", "memory", "interest", "preference", "life_area"]

# goal/memory are per-user nodes (addressed by id); the rest are shared concept nodes
# linked by a per-user relationship (addressed by name).
PER_USER_TYPES = {"goal", "memory"}


class MemoryOp(BaseModel):
    op: Literal["create", "update", "delete"]
    type: MemoryType
    id: str | None = Field(None, description="Existing goal/memory id; required for update/delete of those")
    name: str | None = Field(None, description="Short title (goal/memory) or the value (interest, preference, life area)")
    summary: str | None = Field(None, description="One-sentence detail for goal/memory")
    area: LifeArea | None = Field(None, description="Life area a goal/memory belongs to")
    # Goal-only fields
    timeframe: str | None = Field(None, description="Goals: timeframe as the user said it, e.g. 'next month'")
    target_date: str | None = Field(
        None, pattern=r"^\d{4}-\d{2}(-\d{2})?$", description="Goals: timeframe resolved to YYYY-MM or YYYY-MM-DD"
    )
    target_year: int | None = Field(None, description="Goals: year the goal targets")
    status: Literal["active", "achieved", "abandoned"] | None = Field(None, description="Goals only")


class ExtractionResult(BaseModel):
    profile: ProfileUpdate | None = Field(
        None, description="Only profile facts the user explicitly stated about themselves and that differ from the current profile"
    )
    ops: list[MemoryOp] = Field(default_factory=list, description="Empty when nothing is worth storing")


class FetchArgs(BaseModel):
    types: list[Literal["goal", "memory", "interest", "preference", "life_area"]] | None = Field(
        None, description="Which memory types to fetch; omit for all"
    )
    area: LifeArea | None = Field(None, description="Only goals/memories in this life area")
    include_inactive: bool = Field(False, description="Include achieved/abandoned goals")
