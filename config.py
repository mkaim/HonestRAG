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
    # Prepended to texts before embedding, for models trained with them
    # (e.g. E5: "query: " / "passage: "). Mind the trailing space.
    embed_query_prefix: str = ""
    embed_document_prefix: str = ""
    # Token budgets per chunk, measured with the embedder's tokenizer.
    chunk_tokens: int = 400
    chunk_overlap_tokens: int = 40
    chunk_breadcrumb_tokens: int = 64

    # Any OpenAI-compatible endpoint.
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_structured_output_mode: str = "native"
    llm_native_output_requires_schema_in_instructions: bool = True
    debug: bool = False
