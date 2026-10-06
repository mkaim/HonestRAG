"""Web UI for the RAG corpus.

    uv run uvicorn server:app --reload

Serves a single static page (static/index.html) with a textarea; POSTing a
question to /api/ask streams each pipeline stage as server-sent events,
ending with the verified answer.
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
from rag import Rag

TOP_K = 10

cfg = Settings()
agent = build_agent(
    cfg.llm_base_url,
    cfg.llm_api_key,
    cfg.llm_model,
    structured_output_mode=cfg.llm_structured_output_mode,
    native_output_requires_schema_in_instructions=cfg.llm_native_output_requires_schema_in_instructions,
)

embedder = SenTranEmbedder(
    cfg.embed_model,
    query_prefix=cfg.embed_query_prefix,
    document_prefix=cfg.embed_document_prefix,
)
db = PsqlRagDb(cfg.dsn, embedder)
rag = Rag(db, RRF(), rf_limit=TOP_K)


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
    decomposer = Decomposer(agent)
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

    answerer = Answerer(agent)
    yield _sse(answerer.NAME, "running")
    answered = await answerer.run(prompt_prefix, contents, debug=cfg.debug)
    yield _role_done(answerer, answered)

    # The verifier and completeness checker both judge the answer.
    prompt_prefix += format_answer_suffix(answered.output)

    verifier = Verifier(agent)
    yield _sse(verifier.NAME, "running")
    verified = await verifier.run(prompt_prefix, contents, debug=cfg.debug)
    yield _role_done(verifier, verified)

    checker = CompletenessChecker(agent)
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
