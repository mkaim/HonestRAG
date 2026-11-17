"""Answer-quality eval: the one thing the retrieval eval (golden.py,
test_retrieval_quality.py) doesn't measure - whether the answer/verify
pipeline itself (not just retrieval) does what this project claims. Two
independent measurements:

- test_citation_corruption_is_caught: deterministic, no LLM calls. Takes
  real facts with real quotes from the ingested eval corpus, deliberately
  swaps a sample of their chunk_id/quote to a wrong-but-plausible one, and
  measures what fraction Answerer.verify()'s quote_in_chunk check actually
  catches. This is the project's real safety net - see README's
  "Verification pipeline" section - and until this file it had zero
  measured numbers, only one lucky real-world catch (see git log for the
  CAP-theorem transcript this eval was written in response to).

- test_end_to_end_pipeline_on_gold_questions: slow, needs a real LLM
  endpoint. Runs the actual pipeline (decompose -> search -> answer ->
  verify -> completeness, same calls server.py's _ask_stream makes) over a
  small curated sample of real gold questions, and reports the verifier's
  own stats plus an LLM-judge correctness score against the gold answer.
  Printed, not asserted - same convention as test_retrieval_quality.py.

Run with -s to see output:

    set -a; source .env; set +a
    uv run pytest tests/eval/test_answer_quality.py -s \\
        --embed-model intfloat/multilingual-e5-small
"""

import random

import pytest

from config import Settings
from fusion import RRF
from llm import (
    Answerer,
    CompletenessChecker,
    Decomposer,
    Verifier,
    build_agent,
    chunks_by_id,
    format_answer_suffix,
    format_prompt_prefix,
    format_query_prefix,
    quote_in_chunk,
)
from models import SearchMode
from rag import Rag
from tests.eval.golden import load_gold

# How many of the eval corpus's real chunks to corrupt-and-check per run.
# Large enough for a stable percentage, small enough to stay fast (pure
# code, so this could be much bigger - kept modest since it still has to
# load every chunk's content from the DB first).
CORRUPTION_SAMPLE = 200

# How many real gold questions to run through the full LLM pipeline. Kept
# small deliberately - unlike the retrieval eval, this calls a real LLM
# per question per pipeline stage (4 calls/question), so this is priced
# and timed per run, not a one-time cost like ingestion. Split evenly
# across whatever tiers gold_file has (e.g. SRD's easy/medium) so a tier
# imbalance in the random sample doesn't skew the result - the retrieval
# eval already found medium-tier questions much harder to even retrieve
# for, so a raw unstratified sample could land mostly on one tier by luck.
PIPELINE_SAMPLE = 60

RETRIEVAL_LIMIT = 10


async def test_citation_corruption_is_caught(eval_db, gold_file):
    """Ground truth for the project's core claim ('a verifier is just
    another LLM call and can hallucinate its own supporting quote, so
    nothing is trusted without a deterministic check') - measured by
    deliberately breaking citations and checking the deterministic
    quote_in_chunk layer (Answerer.verify(), same code path the live app
    runs) actually catches them, rather than trusting that it would.

    Pulls real chunks the same way the live pipeline does - via search,
    not direct DB access - using every gold question as a query so the
    pool is large and representative of what actually gets retrieved and
    cited in practice, rather than an arbitrary slice of the corpus."""
    questions = [q.question for q in load_gold(gold_file)]
    rankings = await eval_db.search(SearchMode.SEMANTIC, questions, limit=5)
    chunks = list({r.chunk.id: r for per_query in rankings for r in per_query}.values())
    contents = chunks_by_id(chunks)

    rng = random.Random(0)
    sample = rng.sample(chunks, min(CORRUPTION_SAMPLE, len(chunks)))

    # Two corruption modes, mixed evenly: a plausible real-world mistake
    # each - the first is exactly what the CAP-theorem transcript's real
    # hallucination looked like (right-ish quote, wrong chunk); the second
    # is a quote that doesn't exist verbatim anywhere (garbled/paraphrased).
    caught_wrong_chunk = total_wrong_chunk = 0
    caught_fabricated = total_fabricated = 0
    for i, result in enumerate(sample):
        real_quote = result.chunk.content.split(". ")[0][:80]
        if not real_quote.strip():
            continue
        if i % 2 == 0:
            # Cite a real quote, but against a different chunk than the one
            # it actually came from.
            other = sample[(i + 1) % len(sample)]
            if other.chunk.id == result.chunk.id:
                continue
            verified = quote_in_chunk(real_quote, contents.get(other.chunk.id, ""))
            total_wrong_chunk += 1
            caught_wrong_chunk += not verified
        else:
            # A quote that was never in any chunk verbatim - simulates a
            # paraphrased/fabricated citation.
            fabricated = f"{real_quote[:40]} according to unrelated analysis"
            verified = quote_in_chunk(fabricated, contents.get(result.chunk.id, ""))
            total_fabricated += 1
            caught_fabricated += not verified

    wrong_chunk_rate = caught_wrong_chunk / max(total_wrong_chunk, 1)
    fabricated_rate = caught_fabricated / max(total_fabricated, 1)

    print(
        f"\n--- citation corruption catch rate "
        f"({total_wrong_chunk + total_fabricated} cases) ---"
    )
    print(f"  wrong-chunk citations caught: {caught_wrong_chunk}/{total_wrong_chunk}")
    print(f"  fabricated quotes caught:     {caught_fabricated}/{total_fabricated}")

    # Not a tautology: a wrong-chunk citation can still accidentally verify
    # if the same sentence happens to recur in both chunks (the real
    # glossary/main-text case this eval exists because of) - this measures
    # how often that actually happens on a real corpus, not just that the
    # code path runs.
    assert fabricated_rate == 1.0, "a quote that doesn't exist must never verify"
    assert wrong_chunk_rate > 0.9, (
        f"wrong-chunk catch rate {wrong_chunk_rate:.1%} - unexpectedly low; "
        "check for unusually repetitive text in the corpus"
    )


Tally = dict[str, int]


def _tier_counts(judged_by_tier: dict[str, Tally], tier: str) -> Tally:
    return judged_by_tier.setdefault(tier, {"correct": 0, "partial": 0, "wrong": 0})


def _sample_gold(gold_file, n: int, seed: int = 0):
    """An even split of n across whatever tiers gold_file has (untiered
    datasets - every question's tier is None - get one group, same as
    conftest.py's pytest_generate_tests). An uneven remainder (n not
    divisible by tier count) goes to the earliest tiers alphabetically,
    same tie-break as sorted()."""
    questions = load_gold(gold_file)
    rng = random.Random(seed)
    tiers = sorted({q.tier for q in questions}, key=lambda t: t or "")
    per_tier = n // len(tiers)
    remainder = n % len(tiers)
    sample = []
    for i, tier in enumerate(tiers):
        group = [q for q in questions if q.tier == tier]
        take = per_tier + (1 if i < remainder else 0)
        sample.extend(rng.sample(group, min(take, len(group))))
    rng.shuffle(sample)
    return sample


JUDGE_PROMPT = (
    "You are grading a RAG system's answer against a known-correct gold "
    "answer. Question: {question}\nGold answer: {gold}\nSystem answer: "
    "{answer}\n\nDoes the system answer convey the same facts as the gold "
    "answer (not necessarily the same wording)? Reply with exactly one "
    "word: correct, partial, or wrong."
)


async def test_end_to_end_pipeline_on_gold_questions(eval_db, gold_file):
    """Runs the real pipeline - same stages/calls as server.py's
    _ask_stream - over a small sample of real gold questions, and reports
    both the verifier's own deterministic stats and an LLM-judge
    correctness score against each question's known-correct gold answer.
    No pass/fail threshold, same convention as test_retrieval_quality.py -
    this is a measurement, not a regression gate, until there's a
    baseline run worth comparing against."""
    cfg = Settings()
    try:
        agent = build_agent(cfg.llm_base_url, cfg.llm_api_key, cfg.llm_model)
    except Exception as e:
        pytest.skip(f"no LLM endpoint configured ({e})")

    rag = Rag(eval_db, RRF(), rf_limit=RETRIEVAL_LIMIT)
    sample = _sample_gold(gold_file, PIPELINE_SAMPLE)

    judged = {"correct": 0, "partial": 0, "wrong": 0}
    judged_by_tier: dict[str, dict[str, int]] = {}
    total_facts = total_hallucinated = total_unsupported = total_fully_supported = 0
    errored = 0

    for i, gq in enumerate(sample, start=1):
        tier_label = gq.tier or "untiered"
        print(f"  [{i}/{len(sample)}] ({tier_label}) {gq.question[:80]}", flush=True)
        try:
            decomposer = Decomposer(agent, cfg.llm_simplified_schema_instructions)
            decomposed = await decomposer.run(format_query_prefix(gq.question), {})
            questions = decomposed.output.questions

            results = await rag.search(questions)
            chunks = list(
                {r.chunk.id: r for per_query in results for r in per_query}.values()
            )
            contents = chunks_by_id(chunks)
            if not chunks:
                judged["wrong"] += 1
                _tier_counts(judged_by_tier, tier_label)["wrong"] += 1
                print("      no chunks retrieved - judged wrong", flush=True)
                continue

            prompt_prefix = format_prompt_prefix(gq.question, chunks, questions)

            answerer = Answerer(agent, cfg.llm_simplified_schema_instructions)
            answered = await answerer.run(prompt_prefix, contents)

            prompt_prefix += format_answer_suffix(answered.output)
            verifier = Verifier(agent, cfg.llm_simplified_schema_instructions)
            verified = await verifier.run(prompt_prefix, contents)

            checker = CompletenessChecker(agent, cfg.llm_simplified_schema_instructions)
            await checker.run(prompt_prefix, contents)

            for sub in answered.verification.subanswers:
                for fact in sub.facts:
                    total_facts += 1
                    total_hallucinated += not fact.verified

            total_unsupported += verified.verification.unsupported
            total_fully_supported += verified.verification.fully_supported

            judge_prompt = JUDGE_PROMPT.format(
                question=gq.question, gold=gq.answer, answer=answered.output.answer
            )
            judge_result = await agent.run(judge_prompt)
            verdict = judge_result.output.strip().lower()
            if verdict not in judged:
                verdict = "partial"
            judged[verdict] += 1
            _tier_counts(judged_by_tier, tier_label)[verdict] += 1
            print(f"      judged {verdict}", flush=True)
        except Exception as e:
            # A single LLM call timing out (seen on a long run against a
            # remote endpoint) shouldn't throw away every question already
            # completed - skip it and keep going, report it as excluded
            # from judged's denominator rather than silently miscounted.
            errored += 1
            print(f"      error, skipping: {e}", flush=True)

    n = sum(judged.values())
    print(f"\n--- end-to-end pipeline, {n}/{len(sample)} gold questions judged ---")
    if errored:
        print(f"  ({errored} skipped due to errors, e.g. a timed-out LLM call)")
    for label in ("correct", "partial", "wrong"):
        print(f"  judged {label}: {judged[label]}/{n} ({100 * judged[label] / n:.1f}%)")
    for tier_label, counts in sorted(judged_by_tier.items()):
        tier_n = sum(counts.values())
        breakdown = ", ".join(f"{v}/{tier_n} {k}" for k, v in counts.items())
        print(f"    [{tier_label}] {breakdown}")
    verified_facts = total_facts - total_hallucinated
    print(f"  answerer facts verified: {verified_facts}/{total_facts}")
    print(
        f"  verifier fully supported / unsupported: "
        f"{total_fully_supported} / {total_unsupported}"
    )
