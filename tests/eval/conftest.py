"""Eval environment: ingests a markdown corpus (any directory of .md files,
not just one fixed dataset) once per embedding model into its own Postgres
database (named by model + corpus, so switching either gets its own DB),
then hands tests a ready-to-query Rag/RagDb. Unlike tests/conftest.py's
rag_db (truncated per test, tiny hand-written fixtures), this is for
measuring retrieval/ranking quality on a realistic corpus - slow,
model-dependent, and not part of the default test run.

The DB persists across separate pytest invocations by default (ingest is the
expensive part - a corpus of any real size takes minutes to embed), so
re-running the suite while iterating on chunker/fusion/reranker config is
fast. Pass --drop-eval-db to clean up once you're done, or after changing the
corpus or chunking so stale chunks don't linger.

Corpus and gold set default to the D&D SRD example (see
scripts/fetch_dnd_srd.py to populate it), but any corpus works as long as
it's a directory of .md files plus a gold.json in the generic format
tests/eval/golden.py reads:

    uv run pytest tests/eval --embed-model intfloat/multilingual-e5-small
    uv run pytest tests/eval --corpus-dir tests/eval/data/gutenqa/docs \\
        --gold-file tests/eval/data/gutenqa/gold.json

Not collected by default `pytest` (see pyproject.toml testpaths).
"""

from pathlib import Path

import psycopg
import pytest
import pytest_asyncio
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from chunkers import BlockChunker, FixedSizeChunker
from config import Settings
from db import PsqlRagDb
from embedders import SenTranEmbedder
from loaders import MarkdownFileLoader
from models import Chunk, Chunker, Document

DEFAULT_DATA_DIR = Path(__file__).parent / "data" / "dnd_srd"
_INIT_SQL = (Path(__file__).parent.parent.parent / "postgres_init.sql").read_text()


def pytest_generate_tests(metafunc):
    """Parametrizes `tier` (when a test function takes it) from whatever
    tiers are actually present in --gold-file - [None] (one untiered pass)
    for a dataset that doesn't tag questions by tier, instead of a fixed
    ["easy", "medium"] list that only matches the D&D SRD example."""
    if "tier" not in metafunc.fixturenames:
        return
    from tests.eval.golden import load_gold

    gold_file = Path(metafunc.config.getoption("--gold-file"))
    tiers = sorted({q.tier for q in load_gold(gold_file)}, key=lambda t: t or "")
    metafunc.parametrize("tier", tiers or [None])


def pytest_addoption(parser):
    parser.addoption(
        "--embed-model",
        action="store",
        default=None,
        help="embedding model to evaluate (defaults to RAG_EMBED_MODEL)",
    )
    parser.addoption(
        "--drop-eval-db",
        action="store_true",
        default=False,
        help="drop the eval database at the end of the session instead of "
        "reusing it (and its already-ingested corpus) on the next run",
    )
    parser.addoption(
        "--corpus-dir",
        action="store",
        default=str(DEFAULT_DATA_DIR / "docs"),
        help="directory of .md files to ingest (default: the D&D SRD example)",
    )
    parser.addoption(
        "--gold-file",
        action="store",
        default=str(DEFAULT_DATA_DIR / "gold.json"),
        help="gold Q&A JSON matching --corpus-dir (see tests/eval/golden.py)",
    )
    parser.addoption(
        "--corpus-name",
        action="store",
        default=None,
        help="label distinguishing this corpus in the eval DB name (default: "
        "derived from --corpus-dir's parent directory name)",
    )


def _slug(text: str) -> str:
    return text.rsplit("/", 1)[-1].lower().replace("-", "_").replace(".", "_")


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def embed_model(request, settings: Settings) -> str:
    return request.config.getoption("--embed-model") or settings.embed_model


@pytest.fixture(scope="session")
def corpus_dir(request) -> Path:
    return Path(request.config.getoption("--corpus-dir"))


@pytest.fixture(scope="session")
def gold_file(request) -> Path:
    return Path(request.config.getoption("--gold-file"))


@pytest.fixture(scope="session")
def corpus_name(request, corpus_dir: Path) -> str:
    return request.config.getoption("--corpus-name") or _slug(corpus_dir.parent.name)


def _eval_dsn(
    settings: Settings,
    embed_model: str,
    corpus_name: str,
    variant: str | None = None,
) -> str:
    dbname = conninfo_to_dict(settings.dsn)["dbname"]
    suffix = f"_{variant}" if variant else ""
    return make_conninfo(
        settings.dsn,
        dbname=f"{dbname}_eval_{corpus_name}_{_slug(embed_model)}{suffix}",
    )


@pytest.fixture(scope="session")
def eval_dsn(settings: Settings, embed_model: str, corpus_name: str) -> str:
    return _eval_dsn(settings, embed_model, corpus_name)


@pytest.fixture(scope="session")
def embedder(settings: Settings, embed_model: str) -> SenTranEmbedder:
    return SenTranEmbedder(
        embed_model,
        query_prefix=settings.embed_query_prefix,
        document_prefix=settings.embed_document_prefix,
    )


def _create_eval_db(settings: Settings, eval_dsn: str) -> None:
    admin_conninfo = make_conninfo(settings.dsn, dbname="postgres")
    dbname = conninfo_to_dict(eval_dsn)["dbname"]
    with psycopg.connect(admin_conninfo, autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)
        ).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{dbname}"')

    with psycopg.connect(eval_dsn, autocommit=True) as conn:
        has_schema = conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'document'"
        ).fetchone()
        if not has_schema:
            conn.execute(_INIT_SQL)


def _drop_eval_db(settings: Settings, eval_dsn: str) -> None:
    admin_conninfo = make_conninfo(settings.dsn, dbname="postgres")
    dbname = conninfo_to_dict(eval_dsn)["dbname"]
    with psycopg.connect(admin_conninfo, autocommit=True) as conn:
        conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname = %s AND pid <> pg_backend_pid()",
            (dbname,),
        )
        conn.execute(f'DROP DATABASE IF EXISTS "{dbname}"')


async def _ingest_corpus(db: PsqlRagDb, chunker: Chunker, corpus_dir: Path) -> None:
    loader = MarkdownFileLoader()
    for path in sorted(corpus_dir.glob("*.md")):
        blocks, metadata = loader.load(str(path))
        document = Document(id=path.name, metadata=metadata)
        chunks: list[Chunk] = list(chunker.chunk(document.id, blocks))
        await db.add(document, chunks)


async def _eval_db(
    request,
    settings: Settings,
    eval_dsn: str,
    embedder,
    chunker: Chunker,
    corpus_dir: Path,
):
    """Builds (or reuses) the eval DB for one chunker configuration.
    `eval_dsn` already encodes the embedding model, corpus and chunker
    variant, so each combination gets its own database."""
    if not corpus_dir.exists():
        pytest.skip(
            f"corpus not found at {corpus_dir} - see tests/eval/README.md "
            "or pass --corpus-dir"
        )
    try:
        with psycopg.connect(settings.dsn, connect_timeout=2):
            pass
    except Exception:
        pytest.skip(f"Postgres not reachable at {settings.dsn} (RAG_DSN)")

    _create_eval_db(settings, eval_dsn)

    async with PsqlRagDb(eval_dsn, embedder) as db:
        await db.ensure_vector_table()
        async with db.pool.connection() as conn:
            cur = await conn.execute("SELECT count(*) FROM chunk")
            (count,) = await cur.fetchone()
        if count == 0:
            await _ingest_corpus(db, chunker, corpus_dir)
        yield db

    if request.config.getoption("--drop-eval-db"):
        _drop_eval_db(settings, eval_dsn)


@pytest_asyncio.fixture(scope="session")
async def eval_db(
    request,
    settings: Settings,
    eval_dsn: str,
    embedder: SenTranEmbedder,
    corpus_dir: Path,
):
    """Default chunker config (breadcrumb on, per .env/Settings)."""
    chunker = BlockChunker(
        tokenizer=embedder.tokenizer,
        content_tokens=settings.chunk_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        breadcrumb_tokens=settings.chunk_breadcrumb_tokens,
    )
    async for db in _eval_db(request, settings, eval_dsn, embedder, chunker, corpus_dir):
        yield db


@pytest_asyncio.fixture(scope="session")
async def eval_db_no_breadcrumb(
    request,
    settings: Settings,
    embed_model: str,
    corpus_name: str,
    embedder: SenTranEmbedder,
    corpus_dir: Path,
):
    """Same corpus and chunk boundaries as eval_db, but embedding_text carries
    no [Header > ...] breadcrumb - isolates what the breadcrumb itself is
    worth for semantic search on sibling sections that share wording."""
    dsn = _eval_dsn(settings, embed_model, corpus_name, variant="no_breadcrumb")
    chunker = BlockChunker(
        tokenizer=embedder.tokenizer,
        content_tokens=settings.chunk_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        breadcrumb_tokens=0,
    )
    async for db in _eval_db(request, settings, dsn, embedder, chunker, corpus_dir):
        yield db


@pytest_asyncio.fixture(scope="session")
async def eval_db_fixed_size(
    request,
    settings: Settings,
    embed_model: str,
    corpus_name: str,
    embedder: SenTranEmbedder,
    corpus_dir: Path,
):
    """Naive baseline: FixedSizeChunker instead of BlockChunker - same
    content_tokens/overlap_tokens budget, but no header-awareness and no
    breadcrumb. Isolates what BlockChunker's token-aware, header-aware
    packing is worth over just sliding a fixed-size window."""
    dsn = _eval_dsn(settings, embed_model, corpus_name, variant="fixed_size")
    chunker = FixedSizeChunker(
        tokenizer=embedder.tokenizer,
        content_tokens=settings.chunk_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    async for db in _eval_db(request, settings, dsn, embedder, chunker, corpus_dir):
        yield db
