from dataclasses import asdict, dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from llm.agent import QuoteResult, Role, _chunk_content, quote_in_chunk

VERIFICATION_SUFFIX = (
    "\n\nFor each fact in the answer above, judge whether its quote actually "
    "entails its statement - not just whether the quote is topically "
    "related, but whether a careful reader would agree the quote proves the "
    "statement true. Mark it 'yes' if the quote fully entails the "
    "statement, 'partial' if it only partly supports it, 'no' if it does "
    "not entail it at all, and copy the verbatim span you judged against "
    "(or leave quote empty if 'no'). Separately, list any sentence in the "
    "final answer that is not entailed by any fact above - whether it "
    "contradicts the data or simply adds something no fact supports."
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


class VerificationOutput(BaseModel):
    facts: list[FactCheck]
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description=(
            "sentences in the final answer not entailed by any fact above - "
            "whether they contradict the data or simply add something no "
            "fact supports"
        ),
    )


@dataclass(frozen=True)
class FactCheckResult:
    quote: QuoteResult
    supported: Literal["yes", "partial", "no"]
    """The Verifier's own verdict, downgraded to "no" here if `quote` did
    not verify - an unverifiable quote is not trustworthy support."""

    @property
    def no_claim(self) -> bool:
        """True when there was no quote to check at all - e.g. a fact for a
        sub-question the Answerer correctly declined to answer. Not a
        hallucination or an unsupported claim; nothing was claimed."""
        return not self.quote.chunk_id and not self.quote.quote


@dataclass(frozen=True)
class VerifierVerification:
    facts: list[FactCheckResult] = field(default_factory=list)

    @property
    def hallucinated_quotes(self) -> int:
        return sum(1 for f in self.facts if not f.no_claim and not f.quote.verified)

    @property
    def fully_supported(self) -> int:
        return sum(1 for f in self.facts if f.supported == "yes")

    @property
    def partially_supported(self) -> int:
        return sum(1 for f in self.facts if f.supported == "partial")

    @property
    def unsupported(self) -> int:
        return sum(1 for f in self.facts if not f.no_claim and f.supported == "no")

    @property
    def no_claims(self) -> int:
        return sum(1 for f in self.facts if f.no_claim)

    def to_dict(self) -> dict:
        return {
            "facts": [{**asdict(f), "no_claim": f.no_claim} for f in self.facts],
            "hallucinated_quotes": self.hallucinated_quotes,
            "fully_supported": self.fully_supported,
            "partially_supported": self.partially_supported,
            "unsupported": self.unsupported,
            "no_claims": self.no_claims,
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
            chunk_content = _chunk_content(check.chunk_id, contents)
            verified = quote_in_chunk(check.quote, chunk_content)
            supported = check.supported if verified else "no"
            results.append(
                FactCheckResult(
                    quote=QuoteResult(check.chunk_id, check.quote, verified),
                    supported=supported,
                )
            )
        return VerifierVerification(facts=results)
