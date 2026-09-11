"""Web UI for the RAG corpus.

    uv run uvicorn server:app --reload

Serves a single static page (static/index.html) with a textarea; POSTing a
question to /api/ask retrieves passages and returns the LLM's answer.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import Settings
from db import PsqlRagDb
from embedders import SenTranEmbedder
from fusion import RRF
from llm import answer, build_agent
from rag import Rag

TOP_K = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = Settings()
    embedder = SenTranEmbedder(cfg.embed_model)
    async with PsqlRagDb(cfg.dsn, embedder) as db:
        app.state.rag = Rag(db, RRF(), rf_limit=TOP_K)
        app.state.agent = build_agent(cfg.llm_base_url, cfg.llm_api_key, cfg.llm_model)
        yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.post("/api/ask")
async def ask(req: AskRequest) -> AskResponse:
    [passages] = await app.state.rag.search([req.question])
    if not passages:
        return AskResponse(answer="No matching passages found in the corpus.")

    result = await answer(app.state.agent, req.question, passages)
    return AskResponse(answer=result.text)
