# HonestRAG

A small, self-hosted RAG (retrieval-augmented generation) system built
around one rule: answer only from the retrieved data, and say so plainly
when the data doesn't support an answer, instead of guessing.

Hybrid BM25 + semantic search over Postgres/ParadeDB, fused with RRF,
answered by any OpenAI-compatible LLM (OpenAI, LM Studio, Ollama, etc.).

- **Retrieval**: BM25 (ParadeDB) + semantic (pgvector/HNSW), merged with
  Reciprocal Rank Fusion.
- **Ingestion**: plain text/markdown files, chunked by token count and
  embedded on ingest (see [Chunking](#chunking)).
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

## Embedding model

Any [sentence-transformers](https://sbert.net/) model, set with
`RAG_EMBED_MODEL` (default in `.env.example`:
[intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)).
Some models are trained to see a prefix telling queries and documents apart;
set those with `RAG_EMBED_QUERY_PREFIX` / `RAG_EMBED_DOCUMENT_PREFIX` (E5:
`"query: "` / `"passage: "`, quoted to keep the trailing space), or leave them
empty for models that don't use one.

Each model gets its own vector table, so switching models means re-ingesting.
Changing the prefixes keeps the same table, so re-ingest after that too, or
vectors embedded with and without the prefix end up mixed.

## Chunking

Documents are split into paragraphs, then packed into chunks measured with
the embedding model's own tokenizer. Chunks never span a header boundary, and
paragraphs longer than the chunk budget are split. The stored chunk text is
the plain content; the text that gets embedded additionally carries:

- a `[Title > Section > Subsection]` breadcrumb of the enclosing headers.
  When the path is too long, the innermost header is kept first, then the
  top-level one, then middle levels nearest the innermost; dropped levels and
  cut headers are marked with `…`.
- the tail of the previous chunk, when it's in the same section.

| Variable | Default | Meaning |
|---|---|---|
| `RAG_CHUNK_TOKENS` | 400 | max tokens of stored content per chunk |
| `RAG_CHUNK_OVERLAP_TOKENS` | 40 | tokens carried over from the previous chunk |
| `RAG_CHUNK_BREADCRUMB_TOKENS` | 64 | max tokens of the header breadcrumb |

If content + overlap + breadcrumb would exceed the embedding model's input
limit, the content budget is lowered automatically (with a warning).
Changing these only affects documents ingested afterwards.

## Tests

```bash
set -a; source .env; set +a
uv run pytest
```
