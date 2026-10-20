from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    # Provider names come from app/llm/registry.py
    chat_provider: str = "anthropic"
    chat_model: str = "claude-sonnet-5-5"
    extraction_provider: str = "anthropic"
    extraction_model: str = "claude-haiku-4-5-20251001"
    llm_max_retries: int = 3  # on 429/5xx/network errors, with exponential backoff

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "harness-dev"

    sqlite_path: str = "data/harness.db"

    history_limit: int = 20  # messages (incl. tool rows) loaded per turn
    index_limit: int = 10  # max titles per type in the prompt index
    max_tool_rounds: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
