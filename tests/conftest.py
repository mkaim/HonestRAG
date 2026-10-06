from pathlib import Path

import psycopg
import pytest
import pytest_asyncio
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from transformers import AutoTokenizer

from config import Settings
from db import PsqlRagDb
from embedders import SenTranEmbedder

# Requires RAG_* to be exported (e.g. `set -a; source .env; set +a`) before
# running pytest — see config.py.
_settings = Settings()
EMBED_MODEL = _settings.embed_model

# Tests truncate and reseed their own fixture data, so they must never run
# against RAG_DSN's real database - that would wipe whatever's ingested
# there. TEST_DSN points at a separate "<dbname>_test" database on the same
# Postgres instance instead, created (and schema-initialized) on demand.
_dbname = conninfo_to_dict(_settings.dsn)["dbname"]
TEST_DSN = make_conninfo(_settings.dsn, dbname=f"{_dbname}_test")
_INIT_SQL = (Path(__file__).parent.parent / "postgres_init.sql").read_text()


def _db_reachable() -> bool:
    try:
        with psycopg.connect(_settings.dsn, connect_timeout=2):
            return True
    except Exception:
        return False


def _ensure_test_db() -> None:
    admin_conninfo = make_conninfo(_settings.dsn, dbname="postgres")
    with psycopg.connect(admin_conninfo, autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (f"{_dbname}_test",)
        ).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{_dbname}_test"')

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        has_schema = conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'document'"
        ).fetchone()
        if not has_schema:
            conn.execute(_INIT_SQL)


@pytest.fixture(scope="session")
def hf_tokenizer():
    """Just the embedding model's tokenizer: a few MB, unlike the model."""
    return AutoTokenizer.from_pretrained(EMBED_MODEL)


@pytest.fixture(scope="session")
def embedder() -> SenTranEmbedder:
    return SenTranEmbedder(
        EMBED_MODEL,
        query_prefix=_settings.embed_query_prefix,
        document_prefix=_settings.embed_document_prefix,
    )


@pytest_asyncio.fixture
async def rag_db(embedder):
    if not _db_reachable():
        pytest.skip(f"Postgres not reachable at {_settings.dsn} (RAG_DSN)")

    _ensure_test_db()

    async with PsqlRagDb(TEST_DSN, embedder) as db:
        table = await db.ensure_vector_table()
        async with db.pool.connection() as conn:
            await conn.execute(
                f"TRUNCATE {table}, chunk, document RESTART IDENTITY CASCADE"
            )
        yield db
