"""Retrieval-quality report against whatever gold set --gold-file points at
(the D&D SRD example by default - see tests/eval/README.md).

BM25 and semantic search alone are run only as baselines for comparison -
their absolute scores mostly reflect ParadeDB/pgvector, not this codebase.
What these report is intended to answer is whether *this project's* pieces
earn their keep over those baselines: does RRF fusion beat either mode
alone, does the breadcrumb prefix in embedding_text help semantic search
tell apart near-identical sibling sections (e.g. same stat block field
under different monsters), does a cross-encoder reranking pass improve
ordering. No pass/fail threshold - these print Recall@budget / MRR so you
can diff configs across runs. Run with -s to see the printed report:

    uv run pytest tests/eval -s --embed-model intfloat/multilingual-e5-small

`tier` is parametrized dynamically (see conftest.py's pytest_generate_tests)
from whatever's actually in --gold-file, not a fixed ["easy", "medium"]
list - a dataset without tiers runs once with tier=None.
"""

import time

import pytest

from fusion import RRF
from models import SearchMode
from rag import Rag
from tests.eval.golden import (
    load_gold,
    mean_reciprocal_rank,
    rank_distribution,
    recall_at_budget,
)

# eval_db* fixtures are session-scoped (ingest is too expensive to redo per
# test) and opened on the session event loop (see conftest.py's
# loop_scope="session" fixtures). pytest-asyncio otherwise runs each test
# on its own function-scoped loop by default, which would mismatch the
# fixture's loop - defensive, not a fix for any bug actually seen so far
# (see tests/eval/README.md's PoolTimeout writeup for what the real fix
# was), but a session-scoped async fixture shared across many tests is a
# documented footgun worth avoiding regardless.
pytestmark = pytest.mark.asyncio(loop_scope="session")

# Token budgets to report Recall@ under - same idea as Recall@k, but metered
# by context size (what an LLM would actually be fed) rather than chunk
# count, so configs with different average chunk sizes stay comparable.
BUDGETS = (500, 2000, 4000)

# Retrieved per query before cutting to a token budget - comfortably more
# chunks than the largest budget could ever hold, even for the smallest
# chunker's output, so the budget cutoff is never starved of candidates.
RETRIEVAL_LIMIT = 30


def _report(
    label: str, questions, rankings, tokenizer, elapsed: float | None = None
) -> None:
    print(f"\n--- {label} ({len(questions)} questions) ---")
    for budget in BUDGETS:
        recall = recall_at_budget(questions, rankings, tokenizer, budget)
        print(f"  Recall@{budget}tok: {recall:.3f}")
    print(f"  MRR: {mean_reciprocal_rank(questions, rankings):.3f}")
    if elapsed is not None:
        print(
            f"  search time: {elapsed:.3f}s total, "
            f"{1000 * elapsed / len(questions):.1f}ms/query "
            "(batched - see README for single-query caveat)"
        )


async def test_baseline_single_modes(eval_db, gold_file, tier):
    """Reference points, not the thing under test: RRF fusion and the
    cross-encoder should be judged against these, not in isolation. Also
    where we measure semantic search latency per embedding model - query
    embedding is CPU-bound and model-size-dependent, unlike BM25's lookup."""
    questions = load_gold(gold_file, tier)
    tokenizer = eval_db.embedder.tokenizer
    for mode in (SearchMode.BM25, SearchMode.SEMANTIC):
        start = time.perf_counter()
        rankings = await eval_db.search(
            mode, [q.question for q in questions], limit=RETRIEVAL_LIMIT
        )
        elapsed = time.perf_counter() - start
        _report(
            f"baseline: {mode.value} / {tier}", questions, rankings, tokenizer, elapsed
        )


async def test_fusion_improves_on_single_modes(eval_db, gold_file, tier):
    questions = load_gold(gold_file, tier)
    rag = Rag(eval_db, RRF(), retrieval_limit=50, rf_limit=RETRIEVAL_LIMIT)

    rankings = await rag.search([q.question for q in questions])

    _report(f"RRF-fused / {tier}", questions, rankings, eval_db.embedder.tokenizer)


async def test_breadcrumb_effect_on_semantic_search(
    eval_db, eval_db_no_breadcrumb, gold_file, tier
):
    """Only semantic search is affected: BM25 indexes chunk.content, which
    is identical either way - the breadcrumb only prefixes embedding_text.
    SRD monster/spell entries are full of sibling sections with near-
    identical field text (e.g. a "Restoration" trait reworded per monster),
    so this is where a [Header > ...] breadcrumb should earn its keep."""
    questions = load_gold(gold_file, tier)
    tokenizer = eval_db.embedder.tokenizer

    with_breadcrumb = await eval_db.search(
        SearchMode.SEMANTIC, [q.question for q in questions], limit=RETRIEVAL_LIMIT
    )
    without_breadcrumb = await eval_db_no_breadcrumb.search(
        SearchMode.SEMANTIC, [q.question for q in questions], limit=RETRIEVAL_LIMIT
    )

    _report(
        f"semantic, with breadcrumb / {tier}", questions, with_breadcrumb, tokenizer
    )
    _report(
        f"semantic, no breadcrumb / {tier}", questions, without_breadcrumb, tokenizer
    )


async def test_block_chunker_vs_fixed_size(
    eval_db, eval_db_fixed_size, gold_file, tier
):
    """Unlike the breadcrumb comparison, FixedSizeChunker changes chunk
    boundaries themselves (content, not just embedding_text), so both BM25
    and semantic search are affected here - this isolates what
    BlockChunker's token-aware, header-aware packing is worth over a naive
    fixed-size sliding window with the same token budget. Scored by token
    budget (see golden.recall_at_budget), not chunk count, since
    FixedSizeChunker's chunks run larger on average - comparing a fixed
    chunk count would silently favor whichever config packs more text per
    chunk, independent of ranking quality."""
    questions = load_gold(gold_file, tier)
    tokenizer = eval_db.embedder.tokenizer

    for mode in (SearchMode.BM25, SearchMode.SEMANTIC):
        block_chunked = await eval_db.search(
            mode, [q.question for q in questions], limit=RETRIEVAL_LIMIT
        )
        fixed_size = await eval_db_fixed_size.search(
            mode, [q.question for q in questions], limit=RETRIEVAL_LIMIT
        )
        _report(
            f"{mode.value}, BlockChunker / {tier}", questions, block_chunked, tokenizer
        )
        _report(
            f"{mode.value}, FixedSizeChunker / {tier}",
            questions,
            fixed_size,
            tokenizer,
        )


# How deep to search when characterizing where gold passages actually rank
# (test_semantic_rank_distribution) - deep enough to tell "ranks low" apart
# from "doesn't surface at all", unlike the shallow RETRIEVAL_LIMIT used for
# the budget-based Recall/MRR reports above.
DEEP_RETRIEVAL_LIMIT = 500


async def test_semantic_rank_distribution(eval_db, gold_file, tier):
    """Recall@budget/MRR summarize into one number; this shows the shape of
    the misses instead - whether semantic search is uniformly mediocre or
    bimodal (nails most questions, badly misses a consistent minority).
    Prompted by a live-app case where a textually-relevant, entity-matching
    passage still ranked far outside any reasonable retrieval budget - see
    tests/eval/README.md."""
    questions = load_gold(gold_file, tier)
    rankings = await eval_db.search(
        SearchMode.SEMANTIC, [q.question for q in questions], limit=DEEP_RETRIEVAL_LIMIT
    )
    distribution = rank_distribution(questions, rankings)

    print(f"\n--- semantic rank distribution / {tier} ({len(questions)} questions) ---")
    for label, count in distribution.items():
        print(f"  {label}: {count} ({100 * count / len(questions):.1f}%)")
