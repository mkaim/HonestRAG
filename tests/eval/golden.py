"""Loads a gold Q&A set and scores retrieval against it. Dataset-agnostic:
reads any gold.json in the generic format every scripts/fetch_*.py script
writes to its corpus's data dir -

    [{"question": str, "answer": str, "tier": str | None,
      "passages": [{"content": str, "document": str}]}]

`document` matches the Document.id used at ingest (the markdown filename),
and `content` is gold answer text known to live in that document - exact
offsets aren't required, since chunk boundaries rarely line up with them
anyway; matching by text is more robust.

A retrieved chunk counts as a hit for a gold passage if either text
contains the other once whitespace and quote-mark style are normalized -
e.g. GutenQA's gold text and Gutenberg's own HTML don't always agree on
straight vs curly quotes for the same sentence.

Recall@budget/MRR compare rankings by a *token budget*, not a fixed chunk
count. Comparing "top 10 chunks" across chunking strategies with different
average chunk sizes isn't apples-to-apples - 10 large chunks is a much
bigger slice of context than 10 small ones, so a chunker that merely
produces bigger chunks gets an unfair edge (more total text per chunk =
more likely to contain the gold passage, independent of whether the ranking
actually found it). Spending the same token budget either way - as many
chunks of it as fit, by cumulative size - meters each config by the
resource it would actually cost against an LLM's context window, not by
chunk count.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from models import SearchResult, Tokenizer

_QUOTES_RE = re.compile(r"[\"'“”‘’]")


def _normalize(text: str) -> str:
    text = _QUOTES_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass(frozen=True)
class GoldPassage:
    content: str
    document: str


@dataclass(frozen=True)
class GoldQuestion:
    question: str
    answer: str
    passages: tuple[GoldPassage, ...]
    tier: str | None = None


def load_gold(gold_path: Path, tier: str | None = None) -> list[GoldQuestion]:
    """Loads gold.json. With `tier`, keeps only questions whose own "tier"
    field matches - datasets without tiers (every question's "tier" is
    absent/None) ignore this filter entirely regardless of what's passed."""
    raw = json.loads(Path(gold_path).read_text())
    questions = [
        GoldQuestion(
            question=item["question"],
            answer=item["answer"],
            passages=tuple(
                GoldPassage(p["content"], p["document"]) for p in item["passages"]
            ),
            tier=item.get("tier"),
        )
        for item in raw
    ]
    if tier is not None:
        questions = [q for q in questions if q.tier == tier]
    return questions


def is_hit(passage: GoldPassage, result: SearchResult) -> bool:
    # Suffix match, not equality: a corpus ingested by path (e.g. ingest.py
    # on the live app, which stores the full file path as document_id)
    # still matches gold.json's bare filename.
    if not result.chunk.document_id.endswith(passage.document):
        return False
    gold = _normalize(passage.content)
    chunk = _normalize(result.chunk.content)
    return gold in chunk or chunk in gold


def gold_rank(passage: GoldPassage, ranking: list[SearchResult]) -> int | None:
    """1-based rank of the first result hitting `passage`, or None if absent."""
    for rank, result in enumerate(ranking, start=1):
        if is_hit(passage, result):
            return rank
    return None


def _within_budget(
    ranking: list[SearchResult], tokenizer: Tokenizer, budget_tokens: int
) -> list[SearchResult]:
    """As many leading results as fit within budget_tokens, by cumulative
    embedding_text/content size - never truncates a result, just stops
    before the one that would exceed the budget."""
    kept = []
    spent = 0
    for result in ranking:
        text = result.chunk.embedding_text or result.chunk.content
        cost = tokenizer.count(text)
        if kept and spent + cost > budget_tokens:
            break
        kept.append(result)
        spent += cost
    return kept


def recall_at_budget(
    questions: list[GoldQuestion],
    rankings: list[list[SearchResult]],
    tokenizer: Tokenizer,
    budget_tokens: int,
) -> float:
    """Fraction of gold passages (pooled across all questions) found within
    the leading results of their question's ranking that fit in
    budget_tokens - the token-budget equivalent of Recall@k."""
    hits = total = 0
    for question, ranking in zip(questions, rankings, strict=True):
        within_budget = _within_budget(ranking, tokenizer, budget_tokens)
        for passage in question.passages:
            total += 1
            hits += any(is_hit(passage, r) for r in within_budget)
    return hits / total if total else 0.0


def mean_reciprocal_rank(
    questions: list[GoldQuestion], rankings: list[list[SearchResult]]
) -> float:
    """Per question, 1/rank of its first gold passage found (best case among
    that question's passages), averaged across questions with any hit. Rank
    is a position in the ranking, not a token cost, so this is unaffected by
    chunk size - included alongside recall_at_budget as a size-independent
    signal of how early the right chunk ranks."""
    scores = []
    for question, ranking in zip(questions, rankings, strict=True):
        ranks = [gold_rank(p, ranking) for p in question.passages]
        found = [r for r in ranks if r is not None]
        if found:
            scores.append(1 / min(found))
    return sum(scores) / len(questions) if questions else 0.0


# Bucket boundaries for rank_distribution, paired with their labels - the
# last bucket is "found, but beyond this" and a separate not-found count is
# always reported alongside.
_RANK_BUCKETS = [(10, "1-10"), (30, "11-30"), (100, "31-100")]


def rank_distribution(
    questions: list[GoldQuestion], rankings: list[list[SearchResult]]
) -> dict[str, int]:
    """Where each question's best-ranked gold passage actually lands, in a
    deep ranking (callers should pass a large `limit` to search() - rank
    looks no deeper than what's in `rankings`). Unlike recall_at_budget/MRR,
    which summarize into one number, this shows the shape of the failures:
    whether a model is uniformly mediocre or bimodal (nails most questions,
    badly misses a consistent minority) only shows up in the distribution,
    not the average."""
    buckets = {label: 0 for _, label in _RANK_BUCKETS}
    buckets[f"{_RANK_BUCKETS[-1][0] + 1}+"] = 0
    buckets["not found"] = 0
    for question, ranking in zip(questions, rankings, strict=True):
        ranks = [gold_rank(p, ranking) for p in question.passages]
        found = [r for r in ranks if r is not None]
        if not found:
            buckets["not found"] += 1
            continue
        best = min(found)
        for ceiling, label in _RANK_BUCKETS:
            if best <= ceiling:
                buckets[label] += 1
                break
        else:
            buckets[f"{_RANK_BUCKETS[-1][0] + 1}+"] += 1
    return buckets
