import json
import re
from dataclasses import asdict, dataclass, field
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


@dataclass(frozen=True)
class RoleResult[T: BaseModel]:
    output: T
    verification: object | None = None
    raw_messages: str | None = None


class Role[T: BaseModel]:
    """One stage of the pipeline: a fixed output schema plus a prompt suffix
    appended to a growing prompt_prefix, run through the shared agent."""

    NAME: str
    SUFFIX: str
    Output: type[T]

    def __init__(self, agent: Agent):
        self.agent = agent

    def verify(self, output: T, contents: dict[str, str]) -> object | None:
        """Deterministic, code-level check of `output` against each cited
        chunk_id's actual content, returning this role's own result object.
        Returns None by default; roles that produce quotes override this
        and define their own result dataclass - the LLM's Output schema is
        never mutated, so it stays exactly what the model produced."""
        return None

    async def run(
        self,
        prompt_prefix: str,
        contents: dict[str, str],
        *,
        debug: bool = False,
    ) -> RoleResult[T]:
        prompt = prompt_prefix + self.SUFFIX
        result = await self.agent.run(prompt, output_type=self.Output)
        verification = self.verify(result.output, contents)
        raw_messages = result.new_messages_json().decode() if debug else None
        return RoleResult(
            output=result.output, verification=verification, raw_messages=raw_messages
        )


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


class AnswerOutput(BaseModel):
    facts: list[Fact] = Field(
        description="atomic facts backing the answer; empty if none apply"
    )
    answer: str = Field(description="the synthesized, user-facing answer")


class FactCheck(BaseModel):
    statement: str = Field(description="exact copy of the fact's statement checked")
    supported: Literal["yes", "partial", "no"]
    chunk_id: str = Field(description="chunk id the fact cited")
    quote: str = Field(
        description=(
            "verbatim span from that chunk backing this fact, empty if unsupported"
        )
    )


class VerificationOutput(BaseModel):
    facts: list[FactCheck]
    contradictions: list[str] = Field(
        default_factory=list,
        description="answer claims that conflict with the data",
    )


class SubQuestionCoverage(BaseModel):
    question: str = Field(description="exact copy of the sub-question checked")
    covered: bool = Field(
        description="whether the answer actually addresses this sub-question"
    )
    chunk_id: str = Field(
        default="",
        description="id of the chunk backing coverage, empty if covered is false",
    )
    quote: str = Field(
        default="",
        description=(
            "verbatim span from that chunk backing coverage, empty if covered is false"
        ),
    )


class CompletenessOutput(BaseModel):
    subquestions: list[SubQuestionCoverage] = Field(
        description="coverage of each sub-question the answer was built from"
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


@dataclass(frozen=True)
class QuoteResult:
    """One quote's deterministic verification result: whether it was found,
    verbatim, in the chunk it claims to come from."""

    chunk_id: str
    quote: str
    verified: bool


@dataclass(frozen=True)
class AnswerVerification:
    facts: list[QuoteResult] = field(default_factory=list)

    @property
    def hallucinated_quotes(self) -> int:
        return sum(1 for f in self.facts if not f.verified)

    def to_dict(self) -> dict:
        return {
            "facts": [asdict(f) for f in self.facts],
            "hallucinated_quotes": self.hallucinated_quotes,
        }


class Answerer(Role[AnswerOutput]):
    """Facts each carry a quote; check it against the cited chunk's actual
    text rather than trusting the model's own claim."""

    NAME = "answer"
    SUFFIX = ANSWER_SUFFIX
    Output = AnswerOutput

    def verify(
        self, output: AnswerOutput, contents: dict[str, str]
    ) -> AnswerVerification:
        return AnswerVerification(
            facts=[
                QuoteResult(
                    chunk_id=fact.chunk_id,
                    quote=fact.quote,
                    verified=quote_in_chunk(
                        fact.quote, contents.get(fact.chunk_id, "")
                    ),
                )
                for fact in output.facts
            ]
        )


@dataclass(frozen=True)
class FactCheckResult:
    quote: QuoteResult
    supported: Literal["yes", "partial", "no"]
    """The Verifier's own verdict, downgraded to "no" here if `quote` did
    not verify - an unverifiable quote is not trustworthy support."""


@dataclass(frozen=True)
class VerifierVerification:
    facts: list[FactCheckResult] = field(default_factory=list)

    @property
    def hallucinated_quotes(self) -> int:
        return sum(1 for f in self.facts if not f.quote.verified)

    def to_dict(self) -> dict:
        return {
            "facts": [asdict(f) for f in self.facts],
            "hallucinated_quotes": self.hallucinated_quotes,
        }


class Verifier(Role[VerificationOutput]):
    """Each FactCheck's quote is checked the same way; a yes/partial verdict
    backed by a quote we cannot find is downgraded to "no" here - we do not
    trust the Verifier's own say-so."""

    NAME = "verify"
    SUFFIX = VERIFICATION_SUFFIX
    Output = VerificationOutput

    def verify(
        self, output: VerificationOutput, contents: dict[str, str]
    ) -> VerifierVerification:
        results = []
        for check in output.facts:
            verified = quote_in_chunk(check.quote, contents.get(check.chunk_id, ""))
            supported = check.supported if verified else "no"
            results.append(
                FactCheckResult(
                    quote=QuoteResult(check.chunk_id, check.quote, verified),
                    supported=supported,
                )
            )
        return VerifierVerification(facts=results)


@dataclass(frozen=True)
class SubQuestionResult:
    question: str
    quote: QuoteResult | None
    """None if the model did not claim coverage for this sub-question."""

    @property
    def covered(self) -> bool:
        return self.quote is not None and self.quote.verified


@dataclass(frozen=True)
class CompletenessVerification:
    subquestions: list[SubQuestionResult] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return all(sq.covered for sq in self.subquestions)

    @property
    def missing(self) -> list[str]:
        return [sq.question for sq in self.subquestions if not sq.covered]

    @property
    def hallucinated_quotes(self) -> int:
        return sum(1 for sq in self.subquestions if sq.quote and not sq.quote.verified)

    def to_dict(self) -> dict:
        return {
            "subquestions": [
                {**asdict(sq), "covered": sq.covered} for sq in self.subquestions
            ],
            "complete": self.complete,
            "missing": self.missing,
            "hallucinated_quotes": self.hallucinated_quotes,
        }


class CompletenessChecker(Role[CompletenessOutput]):
    """Each sub-question's coverage claim carries a quote, checked the same
    way as Fact/FactCheck; complete and missing are derived here from
    verified coverage, not the model's own top-line verdict."""

    NAME = "completeness"
    SUFFIX = COMPLETENESS_SUFFIX
    Output = CompletenessOutput

    def verify(
        self, output: CompletenessOutput, contents: dict[str, str]
    ) -> CompletenessVerification:
        results = []
        for sq in output.subquestions:
            quote = None
            if sq.covered:
                verified = quote_in_chunk(sq.quote, contents.get(sq.chunk_id, ""))
                quote = QuoteResult(sq.chunk_id, sq.quote, verified)
            results.append(SubQuestionResult(question=sq.question, quote=quote))
        return CompletenessVerification(subquestions=results)


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
