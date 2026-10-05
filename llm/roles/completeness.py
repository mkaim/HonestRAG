from dataclasses import asdict, dataclass, field

from pydantic import BaseModel, Field

from llm.agent import QuoteResult, Role, _chunk_content, quote_in_chunk

COMPLETENESS_SUFFIX = (
    "\n\nFor each sub-question listed above, decide whether the answer "
    "actually addresses it. If covered, cite the id of the single chunk "
    "backing that coverage plus a verbatim quote copied exactly from that "
    "chunk - do not paraphrase or alter it. If not covered, leave chunk_id "
    "and quote empty."
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
                chunk_content = _chunk_content(sq.chunk_id, contents)
                verified = quote_in_chunk(sq.quote, chunk_content)
                quote = QuoteResult(sq.chunk_id, sq.quote, verified)
            results.append(SubQuestionResult(question=sq.question, quote=quote))
        return CompletenessVerification(subquestions=results)
