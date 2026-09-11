import psycopg
import pytest
import pytest_asyncio

from config import Settings
from db import PsqlRagDb
from embedders import SenTranEmbedder

# Requires RAG_* to be exported (e.g. `set -a; source .env; set +a`) before
# running pytest — see config.py.
_settings = Settings()
TEST_DSN = _settings.dsn
EMBED_MODEL = _settings.embed_model


def _db_reachable() -> bool:
    try:
        with psycopg.connect(TEST_DSN, connect_timeout=2):
            return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def embedder() -> SenTranEmbedder:
    return SenTranEmbedder(EMBED_MODEL)


@pytest_asyncio.fixture
async def rag_db(embedder):
    if not _db_reachable():
        pytest.skip(f"Postgres not reachable at {TEST_DSN} (RAG_DSN)")

    async with PsqlRagDb(TEST_DSN, embedder) as db:
        table = await db.ensure_vector_table()
        async with db.pool.connection() as conn:
            await conn.execute(
                f"TRUNCATE {table}, chunk, document RESTART IDENTITY CASCADE"
            )
        yield db
