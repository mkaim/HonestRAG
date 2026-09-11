import json
import re
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from models import SearchResult

SYSTEM_PROMPT = (
    "You answer questions using only the retrieved context passages you are "
    "given. Never use outside knowledge. If the context does not support an "
    "answer, say so plainly instead of guessing."
)


def _format_context(chunks: list[SearchResult]) -> str:
    return "\n\n".join(f"[{r.chunk.id}] {r.chunk.content}" for r in chunks)


def _normalize(text: str) -> str:
    text = text.strip().strip("\"'“”‘’")
    return re.sub(r"\s+", " ", text).lower()


def quote_in_chunk(quote: str, chunk_content: str) -> bool:
    """Deterministic check that `quote` actually appears in `chunk_content`,
    tolerant of whitespace and quote-mark differences. This is the ground
    truth for whether an LLM-produced quote is real or hallucinated - it
    does not trust the LLM's own supported/quote claim."""
    normalized_quote = _normalize(quote)
    if not normalized_quote:
        return False
    return normalized_quote in _normalize(chunk_content)


def build_agent(
    base_url: str,
    api_key: str,
    model: str,
    *,
    structured_output_mode: str = "native",
    native_output_requires_schema_in_instructions: bool = True,
) -> Agent:
    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    profile = ModelProfile(
        default_structured_output_mode=structured_output_mode,
        native_output_requires_schema_in_instructions=native_output_requires_schema_in_instructions,
    )
    return Agent(
        OpenAIChatModel(model, provider=provider, profile=profile),
        system_prompt=SYSTEM_PROMPT,
    )


class RoleResult[T: BaseModel](BaseModel):
    output: T
    raw_messages: str | None = None


class Role[T: BaseModel]:
    """One stage of the pipeline: a fixed output schema plus a prompt suffix
    appended to a growing prompt_prefix, run through the shared agent."""

    NAME: str
    SUFFIX: str
    Output: type[T]

    def __init__(self, agent: Agent):
        self.agent = agent

    def verify(self, output: T, contents: dict[str, str]) -> None:
        """Deterministic, code-level check applied to `output` in place after
        the model call, given each cited chunk_id's actual content. No-op by
        default; roles that produce quotes override this."""

    async def run(
        self,
        prompt_prefix: str,
        contents: dict[str, str],
        *,
        debug: bool = False,
    ) -> RoleResult[T]:
        prompt = prompt_prefix + self.SUFFIX
        result = await self.agent.run(prompt, output_type=self.Output)
        self.verify(result.output, contents)
        raw_messages = result.new_messages_json().decode() if debug else None
        return RoleResult(output=result.output, raw_messages=raw_messages)


class DecomposeOutput(BaseModel):
    questions: list[str] = Field(
        description="one or more focused sub-questions to retrieve context for"
    )


class Fact(BaseModel):
    chunk_id: str = Field(description="id of the source chunk this fact comes from")
    statement: str = Field(
        description="a single factual statement supported by that chunk"
    )
    quote: str = Field(description="verbatim span from that chunk backing this fact")
    quote_verified: bool = Field(
        default=False,
        description=(
            "set by us, not the model: whether `quote` was deterministically "
            "found in the cited chunk"
        ),
    )


class AnswerOutput(BaseModel):
    facts: list[Fact] = Field(
        description="atomic facts backing the answer; empty if none apply"
    )
    answer: str = Field(description="the synthesized, user-facing answer")
    hallucinated_quotes: int = Field(
        default=0,
        description=(
            "set by us, not the model: how many facts had a quote we could "
            "not find in the cited chunk"
        ),
    )


class FactCheck(BaseModel):
    statement: str = Field(description="exact copy of the fact's statement checked")
    supported: Literal["yes", "partial", "no"]
    chunk_id: str = Field(description="chunk id the fact cited")
    quote: str = Field(
        description=(
            "verbatim span from that chunk backing this fact, empty if unsupported"
        )
    )
    quote_verified: bool = Field(
        default=False,
        description=(
            "set by us, not the model: whether `quote` was deterministically "
            "found in the cited chunk"
        ),
    )


class VerificationOutput(BaseModel):
    facts: list[FactCheck]
    contradictions: list[str] = Field(
        default_factory=list,
        description="answer claims that conflict with the data",
    )
    hallucinated_quotes: int = Field(
        default=0,
        description=(
            "set by us, not the model: how many facts claimed yes/partial "
            "support with a quote we could not find in the cited chunk"
        ),
    )


class SubQuestionCoverage(BaseModel):
    question: str = Field(description="exact copy of the sub-question checked")
    covered: bool = Field(
        description="whether the answer actually addresses this sub-question"
    )
    chunk_id: str = Field(
        default="",
        description=("id of the chunk backing coverage, empty if covered is false"),
    )
    quote: str = Field(
        default="",
        description=(
            "verbatim span from that chunk backing coverage, empty if covered is false"
        ),
    )
    quote_verified: bool = Field(
        default=False,
        description=(
            "set by us, not the model: whether `quote` was deterministically "
            "found in the cited chunk"
        ),
    )


class CompletenessOutput(BaseModel):
    subquestions: list[SubQuestionCoverage] = Field(
        description="coverage of each sub-question the answer was built from"
    )
    complete: bool = Field(
        default=False,
        description=(
            "set by us, not the model: true iff every sub-question is "
            "covered with a verified quote"
        ),
    )
    missing: list[str] = Field(
        default_factory=list,
        description=(
            "set by us, not the model: the question text of every "
            "uncovered sub-question"
        ),
    )
    hallucinated_quotes: int = Field(
        default=0,
        description=(
            "set by us, not the model: how many sub-questions claimed "
            "covered with a quote we could not find in the cited chunk"
        ),
    )


DECOMPOSE_SUFFIX = (
    "\n\nBreak the question above into one or more focused sub-questions "
    "suitable for retrieving supporting passages. If the question is already "
    "focused, return it unchanged as the only entry."
)

ANSWER_SUFFIX = (
    "\n\nUsing only the context passages above, list the atomic facts that "
    "support an answer, each citing the id of the single chunk it came from "
    "plus a verbatim quote copied exactly from that chunk backing it - do "
    "not paraphrase or alter the quote in any way. Then write the final "
    "answer from those facts alone - the answer must not say anything its "
    "facts don't already say. Never state that a source does not mention, "
    "cover, or attribute something merely because no fact says it - only "
    "state an absence if a fact explicitly says the source denies or rules "
    "it out. If the context does not support an answer, say so plainly and "
    "leave facts empty."
)

VERIFICATION_SUFFIX = (
    "\n\nFor each fact in the answer above, check it against the chunk it "
    "cited: mark it 'yes' if the chunk fully supports it, 'partial' if only "
    "partly, 'no' if unsupported, and quote the verbatim span backing it (or "
    "quote empty if unsupported). List separately any claims in the answer "
    "text that contradict the retrieved data."
)

COMPLETENESS_SUFFIX = (
    "\n\nFor each sub-question listed above, decide whether the answer "
    "actually addresses it. If covered, cite the id of the single chunk "
    "backing that coverage plus a verbatim quote copied exactly from that "
    "chunk - do not paraphrase or alter it. If not covered, leave chunk_id "
    "and quote empty."
)


class Decomposer(Role[DecomposeOutput]):
    NAME = "decompose"
    SUFFIX = DECOMPOSE_SUFFIX
    Output = DecomposeOutput


class Answerer(Role[AnswerOutput]):
    """Facts each carry a quote; check it against the cited chunk's actual
    text rather than trusting the model's own claim."""

    NAME = "answer"
    SUFFIX = ANSWER_SUFFIX
    Output = AnswerOutput

    def verify(self, output: AnswerOutput, contents: dict[str, str]) -> None:
        hallucinated = 0
        for fact in output.facts:
            fact.quote_verified = quote_in_chunk(
                fact.quote, contents.get(fact.chunk_id, "")
            )
            if not fact.quote_verified:
                hallucinated += 1
        output.hallucinated_quotes = hallucinated


class Verifier(Role[VerificationOutput]):
    """Each FactCheck's quote is checked the same way; a yes/partial verdict
    backed by a quote we cannot find is downgraded to "no" - we do not trust
    the Verifier's own say-so."""

    NAME = "verify"
    SUFFIX = VERIFICATION_SUFFIX
    Output = VerificationOutput

    def verify(self, output: VerificationOutput, contents: dict[str, str]) -> None:
        hallucinated = 0
        for check in output.facts:
            check.quote_verified = quote_in_chunk(
                check.quote, contents.get(check.chunk_id, "")
            )
            if not check.quote_verified:
                hallucinated += 1
                if check.supported != "no":
                    check.supported = "no"
        output.hallucinated_quotes = hallucinated


class CompletenessChecker(Role[CompletenessOutput]):
    """Each sub-question's coverage claim carries a quote, checked the same
    way as Fact/FactCheck; complete and missing are derived in code from
    verified coverage, not the model's own top-line verdict."""

    NAME = "completeness"
    SUFFIX = COMPLETENESS_SUFFIX
    Output = CompletenessOutput

    def verify(self, output: CompletenessOutput, contents: dict[str, str]) -> None:
        hallucinated = 0
        for sq in output.subquestions:
            if sq.covered:
                sq.quote_verified = quote_in_chunk(
                    sq.quote, contents.get(sq.chunk_id, "")
                )
                if not sq.quote_verified:
                    hallucinated += 1
                    sq.covered = False
        output.hallucinated_quotes = hallucinated
        output.complete = all(sq.covered for sq in output.subquestions)
        output.missing = [sq.question for sq in output.subquestions if not sq.covered]


def format_query_prefix(question: str) -> str:
    return f"Question: {json.dumps(question)}"


def format_prompt_prefix(question: str, chunks: list[SearchResult]) -> str:
    return f"Context:\n{_format_context(chunks)}\n\nQuestion: {json.dumps(question)}"


def format_answer_suffix(answer: AnswerOutput) -> str:
    return f"\n\nAnswer:\n{answer.model_dump_json()}"


def format_subquestions_suffix(questions: list[str]) -> str:
    return f"\n\nSub-questions:\n{json.dumps(questions)}"


def chunks_by_id(chunks: list[SearchResult]) -> dict[str, str]:
    return {r.chunk.id: r.chunk.content for r in chunks}
