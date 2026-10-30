"""Web UI for the RAG corpus.

    uv run uvicorn server:app --reload

Serves a single static page (static/index.html) with a textarea; POSTing a
question to /api/ask streams each pipeline stage as server-sent events,
ending with the verified answer. POSTing a file to /api/documents ingests
it the same way ingest.py's CLI does (see that module for the two modes:
a single file with an optional id, or a .zip of many files with ids
derived from their names) - also streamed as server-sent events, one
running/done (or error) pair per file, so a many-file zip shows progress
and per-file failures instead of one opaque wait.
"""

import json
import tempfile
import zipfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chunkers import BlockChunker
from config import Settings
from db import PsqlRagDb
from embedders import SenTranEmbedder
from fusion import RRF
from llm import (
    Answerer,
    AnswerOutput,
    AnswerVerification,
    CompletenessChecker,
    CompletenessVerification,
    Decomposer,
    Role,
    RoleResult,
    SubQuestionResult,
    Verifier,
    VerifierVerification,
    build_agent,
    chunks_by_id,
    format_answer_suffix,
    format_prompt_prefix,
    format_query_prefix,
)
from loaders import LOADERS
from models import Chunk, Document
from rag import Rag

TOP_K = 10

cfg = Settings()
agent = build_agent(cfg.llm_base_url, cfg.llm_api_key, cfg.llm_model)

embedder = SenTranEmbedder(
    cfg.embed_model,
    query_prefix=cfg.embed_query_prefix,
    document_prefix=cfg.embed_document_prefix,
)
db = PsqlRagDb(cfg.dsn, embedder)
rag = Rag(db, RRF(), rf_limit=TOP_K)
chunker = BlockChunker(
    tokenizer=embedder.tokenizer,
    content_tokens=cfg.chunk_tokens,
    overlap_tokens=cfg.chunk_overlap_tokens,
    breadcrumb_tokens=cfg.chunk_breadcrumb_tokens,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Opens db's connection pool once the event loop is running, and closes
    # it on shutdown. Everything else above is plain sync construction.
    async with db:
        yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class AskRequest(BaseModel):
    question: str


async def _ingest_path(path: Path, doc_id: str) -> int:
    """Loads, chunks and adds one file already on disk, by its real
    extension - same loader lookup ingest.py's CLI uses. Returns the
    chunk count. Raises ValueError if the extension isn't supported, so
    _ingest_stream can report it as this file's error without aborting
    the rest of a multi-file upload."""
    try:
        loader = LOADERS[path.suffix.lower()]
    except KeyError:
        raise ValueError(f"unsupported file type {path.suffix!r}") from None

    blocks, metadata = loader.load(str(path))
    chunks: list[Chunk] = list(chunker.chunk(doc_id, blocks))
    await db.add(Document(id=doc_id, metadata=metadata), chunks)
    return len(chunks)


async def _ingest_stream(
    paths: list[tuple[Path, str]],
) -> AsyncIterator[str]:
    """One running/done (or error) SSE pair per (path, doc_id), so a
    many-file zip shows progress and lets some files fail without losing
    the rest, then a final summary event."""
    ingested: list[str] = []
    failed: list[str] = []
    total_chunks = 0

    for path, doc_id in paths:
        yield _sse(doc_id, "running")
        try:
            chunk_count = await _ingest_path(path, doc_id)
        except Exception as e:
            failed.append(doc_id)
            yield _sse(doc_id, "error", {"error": str(e)})
            continue
        ingested.append(doc_id)
        total_chunks += chunk_count
        yield _sse(doc_id, "done", {"chunks": chunk_count})

    yield _sse(
        "final",
        "done",
        {"documents": ingested, "failed": failed, "chunks": total_chunks},
    )


@app.post("/api/documents")
async def add_document(file: UploadFile, id: str | None = None) -> StreamingResponse:
    """Ingests an uploaded file into the corpus, streaming progress as
    server-sent events (same shape as /api/ask). A single file is one
    document - pass `id` to override the default (the uploaded filename);
    omit it to re-ingest (atomically replace) an existing document of the
    same name. A .zip is treated like ingest.py's --dir: every supported
    file inside becomes its own document, id always the filename - `id`
    isn't valid together with a .zip, since it would collide across files.
    """
    if file.filename is None:
        raise HTTPException(400, "missing filename")

    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)

    if Path(file.filename).suffix.lower() == ".zip":
        if id is not None:
            tmp.cleanup()
            raise HTTPException(400, "id isn't valid for a .zip upload")

        archive_path = tmp_path / "upload.zip"
        archive_path.write_bytes(await file.read())
        try:
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(tmp_path)
        except zipfile.BadZipFile as e:
            tmp.cleanup()
            raise HTTPException(400, "not a valid zip file") from e

        paths = [
            (p, p.name)
            for p in sorted(tmp_path.rglob("*"))
            if p.suffix.lower() in LOADERS and p != archive_path
        ]
        if not paths:
            tmp.cleanup()
            raise HTTPException(400, "zip contains no supported files")
    else:
        path = tmp_path / file.filename
        path.write_bytes(await file.read())
        paths = [(path, id or file.filename)]

    async def stream() -> AsyncIterator[str]:
        with tmp:
            async for event in _ingest_stream(paths):
                yield event

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("static/index.html")


def _sse(stage: str, status: str, data: dict | None = None, debug=None) -> str:
    event = {"stage": stage, "status": status}
    if data is not None:
        event["data"] = data
    if debug is not None:
        event["debug"] = debug
    return f"data: {json.dumps(event)}\n\n"


def _role_done(role: Role, result: RoleResult) -> str:
    data = result.output.model_dump()
    if result.verification is not None:
        data["verification"] = result.verification.to_dict()
    debug = json.loads(result.raw_messages) if result.raw_messages else None
    return _sse(role.NAME, "done", data, debug)


def _subanswers(answer: AnswerOutput, verification: AnswerVerification) -> list[dict]:
    """Each sub-answer with its facts, each fact flagged with whether its
    quote verified against the cited chunk."""
    return [
        {
            "question": sub.question,
            "answer": sub.answer,
            "facts": [
                {
                    "chunk_id": fact.chunk_id,
                    "statement": fact.statement,
                    "quote": fact.quote,
                    "verified": result.verified,
                }
                for fact, result in zip(sub.facts, sub_check.facts, strict=True)
            ],
        }
        for sub, sub_check in zip(
            answer.subanswers, verification.subanswers, strict=True
        )
    ]


def _final(
    answer: str,
    questions: list[str],
    subanswers: list[dict],
    verification: VerifierVerification,
    completeness: CompletenessVerification,
) -> str:
    facts = [fact for sub in subanswers for fact in sub["facts"]]
    stats = {
        "total_facts": len(facts),
        "answer_quotes_verified": sum(fact["verified"] for fact in facts),
        "verifier_fully_supported": verification.fully_supported,
        "verifier_partially_supported": verification.partially_supported,
        "verifier_unsupported": verification.unsupported,
        "verifier_hallucinated_quotes": verification.hallucinated_quotes,
        "verifier_no_claims": verification.no_claims,
        "subquestions_covered": len(completeness.covered),
        "subquestions_not_in_sources": len(completeness.not_in_sources),
        "subquestions_missed": len(completeness.missed),
        "subquestions_total": len(completeness.subquestions),
        "complete": completeness.complete,
    }
    data = {
        "answer": answer,
        "questions": questions,
        "subanswers": subanswers,
        "verification": verification.to_dict(),
        "completeness": completeness.to_dict(),
        "stats": stats,
    }
    return _sse("final", "done", data)


async def _ask_stream(question: str) -> AsyncIterator[str]:
    """Runs the pipeline - decompose, search, answer, verify, completeness -
    streaming a 'running' and a 'done' event per stage, then 'final'."""
    decomposer = Decomposer(agent, cfg.llm_simplified_schema_instructions)
    yield _sse(decomposer.NAME, "running")
    decomposed = await decomposer.run(
        format_query_prefix(question), {}, debug=cfg.debug
    )
    yield _role_done(decomposer, decomposed)
    questions = decomposed.output.questions

    yield _sse("search", "running")
    results = await rag.search(questions)
    # One entry per chunk, even if several sub-questions retrieved it.
    chunks = list({r.chunk.id: r for per_query in results for r in per_query}.values())
    contents = chunks_by_id(chunks)
    debug = {"chunks": contents} if cfg.debug else None
    yield _sse("search", "done", {"chunk_count": len(chunks)}, debug)

    if not chunks:
        not_found = CompletenessVerification(
            [
                SubQuestionResult(q, "not_in_sources", reason="no passages found")
                for q in questions
            ]
        )
        yield _final(
            "No matching passages found in the corpus.",
            questions,
            [],
            VerifierVerification(),
            not_found,
        )
        return

    prompt_prefix = format_prompt_prefix(question, chunks, questions)

    answerer = Answerer(agent, cfg.llm_simplified_schema_instructions)
    yield _sse(answerer.NAME, "running")
    answered = await answerer.run(prompt_prefix, contents, debug=cfg.debug)
    yield _role_done(answerer, answered)

    # The verifier and completeness checker both judge the answer.
    prompt_prefix += format_answer_suffix(answered.output)

    verifier = Verifier(agent, cfg.llm_simplified_schema_instructions)
    yield _sse(verifier.NAME, "running")
    verified = await verifier.run(prompt_prefix, contents, debug=cfg.debug)
    yield _role_done(verifier, verified)

    checker = CompletenessChecker(agent, cfg.llm_simplified_schema_instructions)
    yield _sse(checker.NAME, "running")
    checked = await checker.run(prompt_prefix, contents, debug=cfg.debug)
    yield _role_done(checker, checked)

    yield _final(
        answered.output.answer,
        questions,
        _subanswers(answered.output, answered.verification),
        verified.verification,
        checked.verification,
    )


@app.post("/api/ask")
async def ask(req: AskRequest) -> StreamingResponse:
    return StreamingResponse(_ask_stream(req.question), media_type="text/event-stream")
