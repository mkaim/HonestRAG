from dataclasses import asdict, dataclass, field

from pydantic import BaseModel, Field

from llm.agent import QuoteResult, Role, _chunk_content, quote_in_chunk

ANSWER_SUFFIX = (
    "\n\nUsing only the context passages above, answer each sub-question "
    "listed above in turn. For each one, list the atomic facts that support "
    "its answer, each citing the id of the single chunk it came from plus a "
    "verbatim quote copied exactly from that chunk backing it - do not "
    "paraphrase or alter the quote in any way, and copy one contiguous span; "
    "if the support is in separate sentences, make separate facts rather "
    "than joining them with '...' - then write that "
    "sub-question's own answer from those facts alone. If the question or a "
    "sub-question assumes something the context contradicts - a wrong date, "
    "number, name, or claim - say so explicitly in that sub-answer and give "
    "what the context says, backed by a fact. Never state that a "
    "source does not mention, cover, or attribute something merely because "
    "no fact says it - only state an absence if a fact explicitly says the "
    "source denies or rules it out. If the context does not support an "
    "answer to a sub-question, say so plainly and leave its facts empty. "
    "Finally, write one final answer synthesizing all the sub-answers, "
    "including any corrections - it must not say anything the sub-answers "
    "don't already say."
)


class Fact(BaseModel):
    chunk_id: str = Field(description="id of the source chunk this fact comes from")
    statement: str = Field(
        description="a single factual statement supported by that chunk"
    )
    quote: str = Field(description="verbatim span from that chunk backing this fact")


class SubAnswer(BaseModel):
    question: str = Field(description="exact copy of the sub-question answered")
    facts: list[Fact] = Field(
        description="atomic facts backing this sub-answer; empty if none apply"
    )
    answer: str = Field(description="the answer to just this sub-question")


class AnswerOutput(BaseModel):
    subanswers: list[SubAnswer] = Field(
        description="one answer per sub-question listed above"
    )
    answer: str = Field(
        description="the final, user-facing answer synthesized from the subanswers"
    )


@dataclass(frozen=True)
class SubAnswerVerification:
    question: str
    facts: list[QuoteResult] = field(default_factory=list)

    @property
    def hallucinated_quotes(self) -> int:
        return sum(1 for f in self.facts if not f.verified)


@dataclass(frozen=True)
class AnswerVerification:
    subanswers: list[SubAnswerVerification] = field(default_factory=list)

    @property
    def hallucinated_quotes(self) -> int:
        return sum(sa.hallucinated_quotes for sa in self.subanswers)

    def to_dict(self) -> dict:
        return {
            "subanswers": [
                {
                    "question": sa.question,
                    "facts": [asdict(f) for f in sa.facts],
                    "hallucinated_quotes": sa.hallucinated_quotes,
                }
                for sa in self.subanswers
            ],
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
            subanswers=[
                SubAnswerVerification(
                    question=sub.question,
                    facts=[
                        QuoteResult(
                            chunk_id=fact.chunk_id,
                            quote=fact.quote,
                            verified=quote_in_chunk(
                                fact.quote, _chunk_content(fact.chunk_id, contents)
                            ),
                        )
                        for fact in sub.facts
                    ],
                )
                for sub in output.subanswers
            ]
        )


def format_answer_suffix(answer: AnswerOutput) -> str:
    return f"\n\nAnswer:\n{answer.model_dump_json()}"
