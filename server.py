"""Web UI for the RAG corpus.

    uv run uvicorn server:app --reload

Serves a single static page (static/index.html) with a textarea; POSTing a
question to /api/ask retrieves chunks and returns the LLM's answer.
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
    CompletenessChecker,
    CompletenessVerification,
    Decomposer,
    Role,
    Verifier,
    VerifierVerification,
    build_agent,
    chunks_by_id,
    format_answer_suffix,
    format_prompt_prefix,
    format_query_prefix,
    format_subquestions_suffix,
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

embedder = SenTranEmbedder(cfg.embed_model)
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


def _sse(stage: str, status: str, **fields) -> str:
    return f"data: {json.dumps({'stage': stage, 'status': status, **fields})}\n\n"


class RoleRunner:
    """Wraps role.run() as an async-iterable of SSE lines: a 'running' line,
    then the call, then a 'done' line carrying its output/debug data. Stage
    name comes from the role itself (role.NAME). Iterate it for the SSE
    lines; .output/.verification hold the result once the loop finishes."""

    def __init__(self, role: Role, prompt_prefix: str, contents: dict[str, str]):
        self.role = role
        self.prompt_prefix = prompt_prefix
        self.contents = contents
        self.output = None
        self.verification = None

    async def __aiter__(self) -> AsyncIterator[str]:
        yield _sse(self.role.NAME, "running")

        result = await self.role.run(self.prompt_prefix, self.contents, debug=cfg.debug)
        self.output = result.output
        self.verification = result.verification

        data = result.output.model_dump()
        if result.verification is not None:
            data["verification"] = result.verification.to_dict()
        fields = {"data": data}
        if result.raw_messages is not None:
            fields["debug"] = json.loads(result.raw_messages)
        yield _sse(self.role.NAME, "done", **fields)


async def _ask_stream(question: str) -> AsyncIterator[str]:
    decomposer = RoleRunner(Decomposer(agent), format_query_prefix(question), {})
    async for line in decomposer:
        yield line
    decomposition = decomposer.output

    yield _sse("search", "running")
    results = await rag.search(decomposition.questions)
    by_chunk_id = {r.chunk.id: r for per_query in results for r in per_query}
    chunks = list(by_chunk_id.values())
    yield _sse("search", "done", data={"chunk_count": len(chunks)})

    if not chunks:
        yield _sse(
            "final",
            "done",
            data={
                "answer": "No matching passages found in the corpus.",
                "questions": decomposition.questions,
                "facts": [],
                "verification": VerifierVerification().to_dict(),
                "completeness": CompletenessVerification().to_dict(),
            },
        )
        return

    prompt_prefix = format_prompt_prefix(question, chunks)
    contents = chunks_by_id(chunks)

    answerer = RoleRunner(Answerer(agent), prompt_prefix, contents)
    async for line in answerer:
        yield line
    answer, answer_verification = answerer.output, answerer.verification

    prompt_prefix += format_answer_suffix(answer)

    verifier = RoleRunner(Verifier(agent), prompt_prefix, contents)
    async for line in verifier:
        yield line
    verification = verifier.verification

    completeness_prefix = prompt_prefix + format_subquestions_suffix(
        decomposition.questions
    )

    completeness_checker = RoleRunner(
        CompletenessChecker(agent), completeness_prefix, contents
    )
    async for line in completeness_checker:
        yield line
    completeness = completeness_checker.verification

    facts = [
        {
            "chunk_id": fact.chunk_id,
            "statement": fact.statement,
            "quote": fact.quote,
            "verified": result.verified,
        }
        for fact, result in zip(answer.facts, answer_verification.facts, strict=True)
    ]

    yield _sse(
        "final",
        "done",
        data={
            "answer": answer.answer,
            "questions": decomposition.questions,
            "facts": facts,
            "verification": verification.to_dict(),
            "completeness": completeness.to_dict(),
        },
    )


@app.post("/api/ask")
async def ask(req: AskRequest) -> StreamingResponse:
    return StreamingResponse(_ask_stream(req.question), media_type="text/event-stream")
