from dataclasses import asdict, dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from llm.agent import QuoteResult, Role, _chunk_content, quote_in_chunk

COMPLETENESS_SUFFIX = (
    "\n\nFor each sub-question listed above, set status to one of:\n"
    "- covered: the answer actually and fully addresses it, and a cited "
    "chunk directly entails an answer that matches the question as asked.\n"
    "- not_in_sources: no chunk in the context contains the information, "
    "and the answer says so instead of answering. Check the whole context "
    "before choosing this.\n"
    "- missed: anything else - the answer omits the sub-question, answers a "
    "related but different question, or says the information is missing "
    "although the context contains it.\n"
    "Be strict about the question's specifics: timeframes, locations, "
    "quantities, and named entities must match. If the answer addresses a "
    "related but different question - for example it says 'medieval Europe' "
    "when the question asks about 'modern Europe', or gives a 'what' when "
    "the question asks 'when' - it is missed. If the sub-question rests on a "
    "premise the context contradicts, it is only covered when the answer "
    "points out the error. Explain the status briefly in "
    "reason. If covered, cite the id of the single chunk backing that "
    "coverage plus a verbatim quote copied exactly from that chunk as one "
    "contiguous span - do not paraphrase or alter it. Otherwise leave "
    "chunk_id and quote empty."
)

Status = Literal["covered", "not_in_sources", "missed"]


class SubQuestionCoverage(BaseModel):
    question: str = Field(description="exact copy of the sub-question checked")
    status: Status = Field(
        description=(
            "covered: answered from the context; not_in_sources: the context "
            "lacks it and the answer says so; missed: anything else"
        )
    )
    reason: str = Field(default="", description="briefly why this status")
    chunk_id: str = Field(
        default="",
        description="id of the chunk backing coverage, empty unless covered",
    )
    quote: str = Field(
        default="",
        description="verbatim span from that chunk backing coverage, else empty",
    )


class CompletenessOutput(BaseModel):
    subquestions: list[SubQuestionCoverage] = Field(
        description="coverage of each sub-question the answer was built from"
    )


@dataclass(frozen=True)
class SubQuestionResult:
    question: str
    status: Status
    """covered only with a verified quote; a covered claim whose quote fails
    verification is missed."""
    quote: QuoteResult | None = None
    """None if the model did not claim coverage for this sub-question."""
    reason: str = ""


@dataclass(frozen=True)
class CompletenessVerification:
    subquestions: list[SubQuestionResult] = field(default_factory=list)

    def _with_status(self, status: Status) -> list[str]:
        return [sq.question for sq in self.subquestions if sq.status == status]

    @property
    def covered(self) -> list[str]:
        return self._with_status("covered")

    @property
    def not_in_sources(self) -> list[str]:
        return self._with_status("not_in_sources")

    @property
    def missed(self) -> list[str]:
        return self._with_status("missed")

    @property
    def complete(self) -> bool:
        """Every sub-question is answered or correctly declined as not in the
        sources; only missed ones make the answer incomplete."""
        return not self.missed

    @property
    def hallucinated_quotes(self) -> int:
        return sum(1 for sq in self.subquestions if sq.quote and not sq.quote.verified)

    def to_dict(self) -> dict:
        return {
            "subquestions": [asdict(sq) for sq in self.subquestions],
            "complete": self.complete,
            "not_in_sources": self.not_in_sources,
            "missed": self.missed,
            "hallucinated_quotes": self.hallucinated_quotes,
        }


class CompletenessChecker(Role[CompletenessOutput]):
    """Each sub-question's coverage claim carries a quote, checked the same
    way as Fact/FactCheck; a claim whose quote fails is downgraded to missed.
    not_in_sources is the model's judgement: absence has no quote to check."""

    NAME = "completeness"
    SUFFIX = COMPLETENESS_SUFFIX
    Output = CompletenessOutput

    def verify(
        self, output: CompletenessOutput, contents: dict[str, str]
    ) -> CompletenessVerification:
        results = []
        for sq in output.subquestions:
            status, quote = sq.status, None
            if sq.status == "covered":
                chunk_content = _chunk_content(sq.chunk_id, contents)
                verified = quote_in_chunk(sq.quote, chunk_content)
                quote = QuoteResult(sq.chunk_id, sq.quote, verified)
                status = "covered" if verified else "missed"
            reason = sq.reason
            if status != sq.status:
                reason = f"{reason} [downgraded: quote not found in {sq.chunk_id}]"
            results.append(
                SubQuestionResult(
                    question=sq.question, status=status, quote=quote, reason=reason
                )
            )
        return CompletenessVerification(subquestions=results)
