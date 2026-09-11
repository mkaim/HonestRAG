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
    Decomposer,
    Role,
    Verifier,
    build_agent,
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


async def _run_role(role: Role, prompt_prefix: str) -> tuple:
    result = await role.run(prompt_prefix, debug=cfg.debug)
    fields = {"data": result.output.model_dump()}
    if result.raw_messages is not None:
        fields["debug"] = json.loads(result.raw_messages)
    return result.output, fields


async def _ask_stream(question: str) -> AsyncIterator[str]:
    yield _sse("decompose", "running")
    decomposer = Decomposer(agent)
    decomposition, fields = await _run_role(decomposer, format_query_prefix(question))
    yield _sse("decompose", "done", **fields)

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
                "verification": {"facts": [], "contradictions": []},
                "completeness": {"complete": True, "missing": []},
            },
        )
        return

    prompt_prefix = format_prompt_prefix(question, chunks)

    yield _sse("answer", "running")
    answer, fields = await _run_role(Answerer(agent), prompt_prefix)
    yield _sse("answer", "done", **fields)

    prompt_prefix += format_answer_suffix(answer)

    yield _sse("verify", "running")
    verification, fields = await _run_role(Verifier(agent), prompt_prefix)
    yield _sse("verify", "done", **fields)

    yield _sse("completeness", "running")
    completeness, fields = await _run_role(CompletenessChecker(agent), prompt_prefix)
    yield _sse("completeness", "done", **fields)

    yield _sse(
        "final",
        "done",
        data={
            "answer": answer.answer,
            "questions": decomposition.questions,
            "facts": [f.model_dump() for f in answer.facts],
            "verification": verification.model_dump(),
            "completeness": completeness.model_dump(),
        },
    )


@app.post("/api/ask")
async def ask(req: AskRequest) -> StreamingResponse:
    return StreamingResponse(_ask_stream(req.question), media_type="text/event-stream")
