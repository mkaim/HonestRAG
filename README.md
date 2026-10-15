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

## Verification pipeline

Each question runs through four stages, streamed to the UI as they
complete (`server.py`'s `_ask_stream`):

1. **Decompose** (`llm/roles/decomposer.py`) - the question is split into
   one or more focused sub-questions, each phrased to stand alone (no
   pronouns referring to context outside itself). A multi-part question
   answered and verified as one blob would blur which part is actually
   supported; decomposing keeps grounding precise per sub-question rather
   than per prompt.
2. **Retrieve** (`rag.py`) - every sub-question is searched against both
   BM25 and semantic search, fused with RRF (see
   [Why hybrid BM25 + semantic search](#why-hybrid-bm25--semantic-search)
   below). Chunks are pooled and deduplicated across all sub-questions
   before being handed to the answerer.
3. **Answer** (`llm/roles/answerer.py`) - for each sub-question, the model
   lists atomic facts, each citing the id of the one chunk it came from
   plus a verbatim quote copied from that chunk, then writes that
   sub-question's answer from those facts alone. A final answer
   synthesizes the sub-answers; it is instructed to say nothing the
   sub-answers don't already say. If the context contradicts a premise the
   question assumed, the answerer is told to say so explicitly rather than
   silently answer the assumption.
4. **Verify** (`llm/roles/verifier.py`) - a second, independent LLM pass
   judges each fact: does its quote *entail* the statement (not just
   relate to it), scored yes/partial/no, and separately flags any sentence
   in the final answer not entailed by any fact above (whether it
   contradicts the sources or just adds something unsupported).
5. **Check completeness** (`llm/roles/completeness.py`) - a third
   independent LLM pass judges, per sub-question, whether it's covered,
   correctly declined as not in the sources, or missed - strict about
   matching the question's specifics (timeframes, quantities, named
   entities), so a related-but-different answer counts as missed.

Every stage that cites a quote (answerer, verifier, completeness checker
alike) has that quote checked in code against the chunk it claims to cite
(`llm/agent.py`'s `quote_in_chunk`): normalized for whitespace/quote
marks, tolerant of an ellipsis skipping to a later verbatim span, but
otherwise requiring the quote to actually appear in that chunk's text.
**A verifier is just another LLM call and can hallucinate its own
supporting quote**, so the verifier's "yes" and the completeness
checker's "covered" are themselves downgraded (to "no"/"missed")
whenever their own cited quote fails this check - the system never trusts
an LLM's self-report of its own grounding, including the LLM whose job is
to check grounding.

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

## Design decisions

#### Why hybrid BM25 + semantic search

The two fail in opposite ways. BM25 misses paraphrases and conceptual
matches - ask about "the fee" when the text says "the charge" and a
lexical match finds nothing. Semantic search misses exact literal strings
- an ID, a product code, an uncommon proper name - because an embedding
model generalizes meaning at the cost of exact tokens. For a system that
is built to admit uncertainty rather than guess, relying on only one mode
leaves a blind spot that would surface as "the data doesn't support an
answer" when the data was actually there, just missed by that mode. The
two rankings are merged with Reciprocal Rank Fusion (`fusion/rrf.py`)
rather than picking one.

#### Why decompose into sub-questions before answering

Users often ask multi-part questions in one prompt ("when was X founded
and who led it"). Answering and verifying the whole prompt as one unit
would blur which part is actually grounded - a confident, well-supported
answer to one part and a guess on the other would look like one verified
answer. Decomposing first keeps each claim traceable to its own
sub-question, its own retrieval, and its own citation.

#### Why two separate LLM passes (answer, then verify) instead of one

Built with small, local models (Qwen-class) in mind, not just large
hosted ones. Asking a single pass to both answer a multi-part question
*and* critically judge its own citations in one go asks more reasoning
depth per response than a smaller model reliably gives. Splitting answer
and verification into separate calls keeps each prompt's job narrow -
"answer from these facts" is a different, simpler task than "judge
whether this quote actually entails this statement" - which a smaller
model can do reliably even when it couldn't do both at once.

#### Why the verifier's own citations aren't trusted either

A verifier is just another LLM call, and an LLM call can hallucinate a
quote as easily when checking an answer as when writing one. Honoring the
verifier's "yes" or "covered" verdict just because the model said so would
only move the trust problem one level up, not solve it. So every quote
the verifier (or the completeness checker) cites for its own verdict goes
through the same deterministic `quote_in_chunk` check the answerer's
quotes do - a verdict backed by a quote that doesn't actually appear in
the chunk is downgraded, regardless of how confident the model sounded.

## Known limitations

- **No cross-chunk contradiction detection.** Verification checks a fact
  against the one chunk it cites, not against other retrieved chunks that
  might conflict with it. If two chunks disagree with each other (e.g.
  different documents quoting different figures for the same thing), the
  pipeline has no dedicated mechanism to notice or surface that - at most,
  an LLM judging entailment might incidentally notice something's off, but
  that's not a designed check.
- **No batching or progressive synthesis for very large multi-question
  prompts.** All sub-questions are answered and verified in a single pass
  today. Kept simple deliberately, until that approach actually shows a
  real limit (context size, latency, or quality degrading as sub-question
  count grows) worth solving for.

## How this was built

A self-directed learning project: started solo, for the sake of learning
RAG systems end to end, then extended with an AI coding agent (Claude
Code) under direct supervision for specific pieces - the markdown loader,
the fixed-size-chunker baseline and retrieval eval harness, a couple of
chunker bug fixes - each reviewed and directed step by step rather than
delegated wholesale.

## Tests

```bash
set -a; source .env; set +a
uv run pytest
```
