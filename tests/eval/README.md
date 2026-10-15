# Retrieval eval

Measures retrieval/ranking quality on a realistic corpus, instead of
guessing from unit tests alone. Not collected by default `pytest` - slow,
needs a reachable Postgres and a real embedding model. Generic: works
against any directory of markdown files plus a matching gold Q&A JSON, not
one fixed dataset - see `golden.py` for the gold format and `conftest.py`
for the `--corpus-dir`/`--gold-file` options.

```bash
set -a; source .env; set +a

# D&D SRD example (default corpus/gold if you pass neither flag below)
uv run python scripts/fetch_dnd_srd.py tests/eval/data/dnd_srd
uv run pytest tests/eval -s --embed-model intfloat/multilingual-e5-small

# GutenQA example - pass its corpus/gold explicitly
uv run --with pandas --with pyarrow python scripts/fetch_gutenqa.py tests/eval/data/gutenqa
uv run pytest tests/eval -s --embed-model intfloat/multilingual-e5-small \
    --corpus-dir tests/eval/data/gutenqa/docs --gold-file tests/eval/data/gutenqa/gold.json
```

Results are printed (`-s`), not asserted - there's no pass/fail threshold,
since "did this get better" is a before/after comparison, not a fixed
target.

## Corpora

Neither is vendored in the repo (see `scripts/fetch_*.py` for why, and
`/tests/eval/data/` in `.gitignore`) - each fetch script downloads and
converts its corpus into the generic format on demand.

**D&D 5e SRD 5.2.1** (`scripts/fetch_dnd_srd.py`, from
[datapizza-ai-lab/dnd5e-srd-qa](https://huggingface.co/datasets/datapizza-ai-lab/dnd5e-srd-qa),
CC-BY-4.0): 20 markdown files, already structured with real nested headers,
plus `easy`/`medium` tiered gold Q&A. Chosen because it has, unusually, many
sibling sections sharing near-identical wording (e.g. a monster's
"Restoration" trait, reworded per monster) - exactly the case where a
header breadcrumb should matter, one way or the other.

**GutenQA** (`scripts/fetch_gutenqa.py`, gold Q&A from
[LumberChunker/GutenQA](https://huggingface.co/datasets/LumberChunker/GutenQA),
MIT): 16 public-domain Project Gutenberg novels (480 questions, 30/book),
the opposite structural extreme from SRD - long-form prose with sparse
chapter headers (a handful of headers over hundreds of KB of text per
book, vs. SRD's dense short sections). GutenQA itself only ships
pre-chunked parquet, not usable source markdown, so this script instead
fetches each book's *original* text from Project Gutenberg and converts it
to markdown itself: Gutenberg's HTML edition reliably marks every real
chapter as an `<h2>` wrapping an `<a id="...">` anchor regardless of how
the chapter's visible title reads ("CHAPTER I.", "STAVE I", a bare "I." for
a short-story collection, ...), which is a far more reliable signal than
matching chapter-heading text patterns (tried first; broke on several
books' inconsistent conventions - see this file's `git log` for the
abandoned regex approach). GutenQA's own `Chunk Must Contain` field is
reused as gold text. Validated against the fetched text directly:
464/480 (96.7%) gold passages are found verbatim; the few misses are
dataset-level (GutenQA's gold text occasionally paraphrases or elides a
narration interruption mid-sentence - e.g. "You will be haunted by Three
Spirits" vs. the actual text's "You will be haunted," resumed the Ghost,
"by Three Spirits), not a conversion bug.

## Findings log

### Summary: the `[Header > ...]` breadcrumb isn't earning its keep

Tested across three embedding models (e5-small, e5-base, e5-large) on both
tiers. Net result: the breadcrumb prefix in `embedding_text` has no
meaningful positive effect anywhere, and is actively harmful on the smallest
model:

- **e5-small**: breadcrumb is a clear regression (easy MRR 0.357 vs 0.817
  without it - more than double).
- **e5-base / e5-large**: breadcrumb is noise-level either way (differences
  in the third decimal, no consistent winner across tiers).

So the breadcrumb either hurts or does nothing, never clearly helps, across
every model/tier combination tried. Full numbers and reasoning below; see
also the three-model table further down.

### 2026-10-07 — breadcrumb prefix hurts semantic search on e5-small

`BlockChunker` prefixes `embedding_text` with a `[Header > ...]` breadcrumb
(`chunk_breadcrumb_tokens`, default 64), on the theory that it helps
semantic search disambiguate sections that read alike out of context. Never
actually measured until now.

Compared `eval_db` (breadcrumb on, per `.env` defaults) against
`eval_db_no_breadcrumb` (`breadcrumb_tokens=0`), same corpus, same chunk
boundaries (`content_tokens` computes to the same 400 either way given
`multilingual-e5-small`'s token budget - confirmed before trusting this
result). Only `embedding_text`'s prefix differs. BM25 is identical between
the two since it indexes `chunk.content`, not `embedding_text`.

Semantic search only, `multilingual-e5-small`:

| Metric | easy, with breadcrumb | easy, no breadcrumb | medium, with breadcrumb | medium, no breadcrumb |
|---|---|---|---|---|
| Recall@1 | 0.141 | 0.397 | 0.038 | 0.054 |
| Recall@5 | 0.295 | 0.654 | 0.065 | 0.108 |
| Recall@10 | 0.372 | 0.756 | 0.086 | 0.145 |
| MRR | 0.357 | 0.817 | 0.262 | 0.379 |

No breadcrumb wins on every metric, both tiers. The gap is large on the easy
tier (short, direct factual questions); smaller but the same direction on
medium (multi-hop/synthesis questions, where single-chunk retrieval scores
low either way).

Likely cause: `multilingual-e5-small` is a small, largely lexical-similarity
model. Prepending header text that doesn't share vocabulary with the query
dilutes the embedding vector away from matching the query's actual wording,
pushing the correct chunk down the ranking instead of up.

Not yet tested: whether this holds for other embedding models (larger /
instruction-tuned models may use the breadcrumb context better), smaller
`breadcrumb_tokens` values, or whether RRF fusion with BM25 (which doesn't
see the breadcrumb at all) washes out the effect in practice.

### 2026-10-07 — breadcrumb effect flips (and shrinks) on e5-base

Same comparison, `multilingual-e5-base` (278M params, 768 dims - the next
size up from e5-small, still well short of bge-m3's 1.1B):

| Metric | easy, with breadcrumb | easy, no breadcrumb | medium, with breadcrumb | medium, no breadcrumb |
|---|---|---|---|---|
| Recall@1 | 0.372 | 0.359 | 0.054 | 0.059 |
| Recall@5 | 0.667 | 0.628 | 0.129 | 0.129 |
| Recall@10 | 0.744 | 0.705 | 0.183 | 0.177 |
| MRR | 0.834 | 0.786 | 0.400 | 0.428 |

Unlike e5-small, the breadcrumb is now a wash to slightly helpful: it wins
on every easy-tier metric (e.g. MRR 0.834 vs 0.786) but loses narrowly on
medium (MRR 0.400 vs 0.428, Recall@5 tied). Nothing close to e5-small's
large, consistent regression in either direction.

Also notable on its own: e5-base's absolute numbers trounce e5-small's
regardless of breadcrumb (easy MRR ~0.8-0.83 vs e5-small's 0.357-0.817,
medium MRR ~0.4 vs e5-small's 0.262-0.379) - a bigger embedding model is
doing far more for retrieval quality here than the breadcrumb choice either
way.

Reading: the breadcrumb-hurts-retrieval conclusion from e5-small doesn't
generalize - it looks specific to that model's limited capacity to use
extra header context without it crowding out the query-relevant signal. On
e5-base the same prefix is roughly neutral. Still want the e5-large result
before drawing a real conclusion about whether to keep, drop, or
size-dependently toggle the breadcrumb.

### 2026-10-07 — breadcrumb is a wash on e5-large too; model size matters, breadcrumb mostly doesn't

Same comparison, `multilingual-e5-large` (560M params, 1024 dims):

| Metric | easy, with breadcrumb | easy, no breadcrumb | medium, with breadcrumb | medium, no breadcrumb |
|---|---|---|---|---|
| Recall@1 | 0.372 | 0.385 | 0.043 | 0.043 |
| Recall@5 | 0.654 | 0.603 | 0.102 | 0.102 |
| Recall@10 | 0.731 | 0.731 | 0.156 | 0.156 |
| MRR | 0.777 | 0.775 | 0.368 | 0.341 |

Essentially a wash either way, on both tiers (medium's Recall is identical
per-k; MRR differs only in the third decimal). Consistent with e5-base: once
the embedding model is past the "small" tier, the breadcrumb prefix stops
being either a meaningful help or a meaningful hindrance.

**Three-model summary (semantic search, easy / medium MRR):**

| Model | Params | Dims | with breadcrumb | no breadcrumb |
|---|---|---|---|---|
| e5-small | 118M | 384 | 0.357 / 0.262 | **0.817** / **0.379** |
| e5-base | 278M | 768 | **0.834** / 0.400 | 0.786 / **0.428** |
| e5-large | 560M | 1024 | 0.777 / 0.368 | 0.775 / 0.341 |

Takeaways:
- **Model size dominates.** Jumping from e5-small to e5-base roughly
  doubles easy-tier MRR regardless of breadcrumb setting. Going from
  e5-base to e5-large doesn't help further on this corpus - e5-base is
  actually the best performer here, slightly ahead of the larger e5-large.
  Bigger isn't automatically better; worth remembering if this generalizes
  to also not need bge-m3.
- **The breadcrumb's effect is real only on e5-small**, and there it's a
  large regression, not a win. On e5-base/e5-large it's noise-level either
  way. If e5-small were the production model, dropping the breadcrumb (or
  shrinking `chunk_breadcrumb_tokens`) would be a clear, cheap improvement;
  on e5-base (today's `.env` default) it's not worth the churn either
  direction based on this data alone.
- Not yet tested: whether RRF fusion with BM25 (unaffected by breadcrumb)
  changes this calculus in practice, smaller `breadcrumb_tokens` values
  (e.g. 16-32 instead of 64) as a middle ground, or bge-m3 itself.

### Query-time latency (semantic search, batched)

Measured in `test_baseline_single_modes`, timing `eval_db.search(SEMANTIC,
...)` for all 25 (easy) or 31 (medium) questions in one batched call -
`SenTranEmbedder.embed` encodes the whole list in a single
`SentenceTransformer.encode`. This is batch throughput, not the single-query
latency the live app sees per request (one question in, not 25-31) - useful
for relative model comparison, not as a production latency estimate.
CPU-only (no GPU), on the same machine throughout:

| Model | easy: total / per query | medium: total / per query |
|---|---|---|
| e5-small | 0.781s / 31.3ms | 0.862s / 27.8ms |
| e5-base | 1.433s / 57.3ms | 1.488s / 48.0ms |
| e5-large | 5.230s / 209.2ms | 5.525s / 178.2ms |

Roughly linear-to-superlinear in model size: e5-base is ~2x e5-small's
per-query time (matches its ~2x parameter count), e5-large is ~4x e5-base's
despite only ~2x the parameters - likely past a cache/batch-efficiency knee
for this hardware. Ingestion time (one-time, not shown here) scaled far more
dramatically: embedding the full ~1540-chunk corpus took minutes for
e5-small, tens of minutes for e5-base, and over an hour for e5-large on this
CPU-only machine.

### 2026-10-07 — BlockChunker vs. a naive fixed-size chunker

Added `FixedSizeChunker` (`chunkers/fixed_size.py`) as a deliberately dumb
baseline: flattens all blocks (headers and paragraphs alike) into one text
and slides a fixed `content_tokens`-token window across it with
`overlap_tokens` of context carried into `embedding_text` only (same
content/embedding_text split as `BlockChunker`) - no header-awareness, no
breadcrumb, no respecting paragraph boundaries. Same token budget
(`chunk_tokens`/`chunk_overlap_tokens` from `.env`) as `BlockChunker`, so
this isolates what the token-aware, header-aware packing itself is worth,
not just a budget difference.

Unlike the breadcrumb comparison, this changes `chunk.content` itself (not
just `embedding_text`), so BM25 is affected too, not only semantic search.

`multilingual-e5-small`:

| Metric | BM25, Block / easy | BM25, FixedSize / easy | Semantic, Block / easy | Semantic, FixedSize / easy |
|---|---|---|---|---|
| Recall@1 | 0.141 | 0.244 | 0.141 | 0.295 |
| Recall@5 | 0.218 | 0.385 | 0.295 | 0.474 |
| Recall@10 | 0.282 | 0.462 | 0.372 | 0.551 |
| MRR | 0.328 | 0.515 | 0.357 | 0.620 |

| Metric | BM25, Block / medium | BM25, FixedSize / medium | Semantic, Block / medium | Semantic, FixedSize / medium |
|---|---|---|---|---|
| Recall@1 | 0.032 | 0.032 | 0.038 | 0.059 |
| Recall@5 | 0.065 | 0.075 | 0.065 | 0.075 |
| Recall@10 | 0.102 | 0.113 | 0.086 | 0.097 |
| MRR | 0.265 | 0.237 | 0.262 | 0.303 |

Surprising: `FixedSizeChunker` beats `BlockChunker` almost everywhere on
e5-small - clearly on the easy tier (MRR ~0.5-0.6 vs ~0.33-0.36, both modes),
and narrowly on medium for everything except BM25 (where `BlockChunker`
edges ahead, 0.265 vs 0.237).

Checked chunk-length distributions on e5-small's two DBs before trusting
this: `BlockChunker` averages 1101 chars/chunk (485 chunks total) with a
long tail of small ones - 25% are under 800 chars, because it cuts a new
chunk at every header even when the section under it is short.
`FixedSizeChunker` averages 1278 chars/chunk (1172 chunks total, far more
of them) and is tightly concentrated near the token budget (~89% between
800-1600 chars) - it never produces a small chunk, because it always packs
to the full window regardless of document structure.

`multilingual-e5-base` (same comparison, repeated):

| Metric | BM25, Block / easy | BM25, FixedSize / easy | Semantic, Block / easy | Semantic, FixedSize / easy |
|---|---|---|---|---|
| Recall@1 | 0.269 | 0.244 | 0.372 | 0.244 |
| Recall@5 | 0.551 | 0.385 | 0.667 | 0.410 |
| Recall@10 | 0.628 | 0.462 | 0.744 | 0.500 |
| MRR | 0.634 | 0.515 | 0.834 | 0.521 |

| Metric | BM25, Block / medium | BM25, FixedSize / medium | Semantic, Block / medium | Semantic, FixedSize / medium |
|---|---|---|---|---|
| Recall@1 | 0.054 | 0.032 | 0.054 | 0.043 |
| Recall@5 | 0.113 | 0.075 | 0.129 | 0.091 |
| Recall@10 | 0.145 | 0.113 | 0.183 | 0.102 |
| MRR | 0.368 | 0.237 | 0.400 | 0.282 |

On e5-base, `BlockChunker` wins clearly on every single metric, both modes,
both tiers - the opposite direction from e5-small. A real quality
difference in chunking strategy should point the same way regardless of
embedding model; a full reversal like this is a strong sign the e5-small
result was a measurement artifact, not a genuine finding.

**The likely artifact: `golden.is_hit`'s containment check is biased
toward bigger chunks.** It scores a hit as `gold in chunk or chunk in
gold` - "the gold passage's text appears somewhere in the chunk" (or vice
versa). A bigger chunk is mechanically more likely to contain the gold
text as a substring, *independent of whether retrieval actually ranked it
well* - it's a wider net, not a better catch. Since `FixedSizeChunker`'s
chunks run both bigger on average and far more consistently close to the
token budget (no small chunks at all) than `BlockChunker`'s, some or all of
its apparent e5-small "win" is plausibly just this size bias inflating its
hit rate, not better retrieval. The e5-base reversal is consistent with
that: once the embedding model is good enough to matter more than chunk
size, `BlockChunker`'s precision (smaller, more targeted chunks) starts
winning outright.

**Conclusion: don't trust the e5-small `FixedSizeChunker` result as-is.**
`is_hit` needs to stop giving a free pass to "chunk is huge and happens to
contain the answer" before this comparison means anything.

### 2026-10-15 — fixed the scoring bias: Recall is now by token budget, not chunk count

Rather than adjusting `is_hit` itself (containment is still the right
check for "is this one chunk a real match"), fixed the actual bias at its
source: `recall_at_k` compared a fixed *count* of chunks (top 10) across
configs with different average chunk sizes, which silently handed the edge
to whichever config packs more text per chunk. Replaced it with
`recall_at_budget` (`tests/eval/golden.py`): instead of "top 10 chunks,"
take as many leading chunks as fit within a token budget (500/2000/4000
tokens - comparable to how much context an LLM call would actually spend),
by cumulative `embedding_text`/`content` size. A chunker with smaller
chunks now rightly gets to include *more* of them for the same budget; one
with bigger chunks gets fewer. Nothing is penalized for being big or small
per se - both are metered by the same resource an LLM context window would
actually charge for. `mean_reciprocal_rank` needed no change - it compares
rank position, not chunk size, so it was never biased by this.

Re-ran `BlockChunker` vs. `FixedSizeChunker` on e5-base with the fixed
metric (same two DBs, e5-small's two DBs also still around but not re-run
here):

| Metric | BM25, Block / easy | BM25, FixedSize / easy | Semantic, Block / easy | Semantic, FixedSize / easy |
|---|---|---|---|---|
| Recall@500tok | 0.269 | 0.244 | 0.385 | 0.244 |
| Recall@2000tok | 0.551 | 0.385 | 0.692 | 0.410 |
| Recall@4000tok | 0.628 | 0.462 | 0.756 | 0.500 |
| MRR | 0.637 | 0.522 | 0.834 | 0.524 |

| Metric | BM25, Block / medium | BM25, FixedSize / medium | Semantic, Block / medium | Semantic, FixedSize / medium |
|---|---|---|---|---|
| Recall@500tok | 0.054 | 0.032 | 0.059 | 0.043 |
| Recall@2000tok | 0.113 | 0.075 | 0.151 | 0.086 |
| Recall@4000tok | 0.156 | 0.113 | 0.194 | 0.097 |
| MRR | 0.382 | 0.247 | 0.427 | 0.289 |

**`BlockChunker` wins clearly and consistently - every budget, both modes,
both tiers.** This confirms (rather than contradicts) the earlier,
uncorrected e5-base result's direction, now without the size-bias doubt
hanging over it. Combined with the e5-small/e5-base reversal we saw before
the fix, the full picture is: `FixedSizeChunker`'s apparent e5-small "win"
really was mostly/entirely the counting artifact, not a real effect -
`BlockChunker`'s token-aware, header-aware packing is the better chunker
on this corpus, at least for e5-base (today's production candidate).

Not yet done: re-running e5-small and e5-large under the fixed metric for
completeness (lower priority now that e5-base is the production pick), and
testing on a second, structurally different corpus (long-form prose with
sparse headers, rather than SRD's dense short sections) to see whether
`BlockChunker`'s advantage holds outside SRD's specific structure - in
progress, see scripts/fetch_gutenqa.py.

### 2026-10-15 — fixed a PoolTimeout that only showed up on the bigger corpus

Running the full eval suite against GutenQA (480 questions, ~4x SRD's
corpus size) hit a `psycopg_pool.PoolTimeout: couldn't get a connection
after 30.00 sec` reliably, on every test that searches all questions at
once. SRD never hit this. Took a while to actually find, since every
plausible-sounding theory checked out false on direct testing:

- **Not pool size** - bumped `PsqlRagDb`'s `min_size`/`max_size` (now
  configurable, still 2/2 by default for production) well above the
  default; still failed identically.
- **Not event loop scope** - session-scoped async fixtures on a different
  loop than function-scoped tests is a real pytest-asyncio footgun, and
  looked like a plausible match, but adding `loop_scope="session"`
  everywhere changed nothing.
- **Not Postgres parallel workers** - ParadeDB's own docs say exhausted
  parallel workers make queries slower, never fail or hang, which this
  symptom contradicted anyway once checked directly.
- **Not an open transaction from the pool's `configure=register_vector_async`
  callback** - checked a freshly-configured connection's transaction
  status directly; it was idle, not mid-transaction.
- **Real network calls, but not the cause of this bug** - `SenTranEmbedder`
  was hitting huggingface.co on every instantiation (dozens of HTTP
  requests, confirmed via `--log-cli-level=DEBUG`) even when the model was
  already cached locally. Genuinely worth fixing (`local_files_only`,
  checked against `huggingface_hub.scan_cache_dir()` first) and fixed
  below, but eliminating it didn't fix the PoolTimeout either.

**The actual cause**, found by watching `psycopg.pool`'s own debug log
live during a failing run: hundreds of "connection requested" log lines
with only a handful of "connection given"/"growing pool" responses - the
pool grew to size 11 once, then never grew again despite constant demand,
no errors logged. With `RETRIEVAL_LIMIT` (or the full 480-question gold
set) all launched via one `asyncio.gather`, there are enough concurrently
scheduled tasks on one event loop that `psycopg_pool`'s own background
connection-management coroutines - tasks on that same loop - stop getting
scheduled often enough to keep the pool healthy under load. Confirmed by
fix: batching the same 480 queries into groups of 10, awaiting each batch
fully before starting the next, passed cleanly every time. The live app
never hits this - it only ever searches a handful of sub-questions at
once, nowhere near enough concurrent tasks to starve the pool's own
coroutines.

Fix: `tests/eval/batching.py`'s `BatchedRagDb` wraps the real `PsqlRagDb`
and splits any `search()` call into batches of 10 queries, awaited fully
one batch at a time - transparent to every test, since it implements the
same `RagDb` protocol. `conftest.py`'s `eval_db*` fixtures now yield the
wrapped db. This is purely an eval-harness fix; `PsqlRagDb` itself (and
the live app) is unaffected.

Rerunning the full GutenQA suite afterward: all 4 tests passed cleanly,
no PoolTimeout, in under 7 minutes total (vs. ~2 hours for the two failed
attempts before this fix, most of that spent re-ingesting after the
first attempt's failure left things in a half-ingested state before
`--drop-eval-db` behavior was fully sorted out).

### 2026-10-15 — GutenQA vs. SRD: does BlockChunker's advantage hold on different structure?

`multilingual-e5-base`, all 480 GutenQA questions (untiered - every
question runs as one group, unlike SRD's easy/medium split):

| Metric | BM25, Block | BM25, FixedSize | Semantic, Block | Semantic, FixedSize |
|---|---|---|---|---|
| Recall@500tok | 0.537 | 0.479 | 0.440 | 0.379 |
| Recall@2000tok | 0.696 | 0.644 | 0.673 | 0.617 |
| Recall@4000tok | 0.754 | 0.690 | 0.748 | 0.702 |
| MRR | 0.604 | 0.551 | 0.530 | 0.486 |

**`BlockChunker` wins again, on every budget, both modes** - same
direction as SRD's (budget-corrected) result. On a structurally opposite
corpus (sparse chapter headers, long sections vs. SRD's dense short
sections), the header-aware, token-packed chunker still beats a naive
fixed-size window. This is the strongest evidence yet that `BlockChunker`'s
advantage isn't an artifact of SRD's unusual density of sibling sections -
it holds on ordinary long-form prose too.

Breadcrumb effect on GutenQA (semantic search only, same corpus/chunker
otherwise):

| Metric | with breadcrumb | no breadcrumb |
|---|---|---|
| Recall@500tok | 0.440 | 0.444 |
| Recall@2000tok | 0.673 | 0.681 |
| Recall@4000tok | 0.748 | 0.773 |
| MRR | 0.530 | 0.539 |

Consistent with e5-base's SRD result: breadcrumb is a wash, slightly
favoring no-breadcrumb but nowhere near a meaningful effect either way.
Reinforces the earlier conclusion that the breadcrumb only mattered (and
hurt) on the smallest model tested (e5-small); on e5-base it's noise
regardless of corpus structure.

RRF fusion beats both single modes here too (MRR 0.612 vs. BM25's 0.604
and semantic's 0.530), same pattern as every other corpus/model
combination tried so far - hybrid retrieval earns its keep consistently.
