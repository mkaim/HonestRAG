# HonestRAG

A small, self-hosted RAG (retrieval-augmented generation) system built
around one rule: answer only from the retrieved data, and say so plainly
when the data doesn't support an answer, instead of guessing.

Hybrid BM25 + semantic search over Postgres/ParadeDB, fused with RRF,
answered by any OpenAI-compatible LLM (OpenAI, LM Studio, Ollama, etc.).

- **Retrieval**: BM25 (ParadeDB) + semantic (pgvector/HNSW), merged with
  Reciprocal Rank Fusion.
- **Ingestion**: plain text/markdown files, chunked and embedded on ingest.
- **Answering**: retrieved passages are handed to an LLM via
  [pydantic-ai](https://ai.pydantic.dev/), against any OpenAI-compatible
  endpoint — configure `RAG_LLM_BASE_URL`/`RAG_LLM_API_KEY`/`RAG_LLM_MODEL`.
- **Access**: a single-page web UI, served by FastAPI.

## Requirements

- Docker (and Docker Compose)
- [uv](https://docs.astral.sh/uv/) — for running ingestion and tests on the host
- An OpenAI-compatible LLM endpoint reachable from your machine

## Setup

```bash
cp .env.example .env   # edit as needed
docker compose up -d   # starts Postgres + the web UI
```

`docker compose up -d` starts two containers: `postgres` (ParadeDB, with the
schema in `postgres_init.sql` applied automatically on first boot) and `web`
(the FastAPI app). The web UI is served at http://localhost:8000 (or
`$WEB_PORT`) once both are up and healthy.

## Usage

Ask questions via the web UI (http://localhost:8000). Ingestion runs on the
host, so load the env vars first:

```bash
set -a; source .env; set +a

uv run python ingest.py --dir ~/notes/       # ingest documents
```

## Tests

```bash
set -a; source .env; set +a
uv run pytest
```
