from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from the environment (RAG_* vars).

    No defaults and no .env autoloading here — export the vars yourself, or
    load them explicitly before running, e.g.:

        set -a; source .env; set +a; uv run python ingest.py "..."
    """

    model_config = SettingsConfigDict(env_prefix="RAG_")

    dsn: str
    embed_model: str

    # Any OpenAI-compatible endpoint.
    llm_base_url: str
    llm_api_key: str
    llm_model: str
