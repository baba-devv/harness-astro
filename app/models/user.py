from datetime import date, time

from pydantic import BaseModel, Field, field_validator

# Client-supplied ids (e.g. "user-123") are allowed; keep them URL/log safe
ID_PATTERN = r"^[A-Za-z0-9_.:-]{1,64}$"


class ProfileFields(BaseModel):
    """Every field is optional: users can start with nothing and fill in details in chat."""

    name: str | None = Field(None, min_length=1, max_length=100)
    date_of_birth: date | None = None
    time_of_birth: time | None = None
    birth_place: str | None = Field(None, min_length=1, max_length=100)

    @field_validator("date_of_birth")
    @classmethod
    def not_in_future(cls, v: date | None) -> date | None:
        if v and v > date.today():
            raise ValueError("date_of_birth cannot be in the future")
        return v


class UserCreate(ProfileFields):
    id: str | None = Field(None, pattern=ID_PATTERN, description="Optional; generated if omitted")
    preferred_language: str = Field("English", min_length=1, max_length=40)


class ProfileUpdate(ProfileFields):
    """Partial update: only non-null fields change."""

    preferred_language: str | None = Field(None, min_length=1, max_length=40)


class User(ProfileFields):
    id: str
    preferred_language: str = "English"
    sun_sign: str | None = None
