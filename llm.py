import json
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

    async def run(self, prompt_prefix: str, *, debug: bool = False) -> RoleResult[T]:
        prompt = prompt_prefix + self.SUFFIX
        result = await self.agent.run(prompt, output_type=self.Output)
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


class AnswerOutput(BaseModel):
    facts: list[Fact] = Field(
        description="atomic facts backing the answer; empty if none apply"
    )
    answer: str = Field(description="the synthesized, user-facing answer")


class FactCheck(BaseModel):
    statement: str = Field(description="exact copy of the fact's statement checked")
    chunk_id: str = Field(description="chunk id the fact cited")
    supported: Literal["yes", "partial", "no"]
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


class CompletenessOutput(BaseModel):
    complete: bool
    missing: list[str] = Field(
        default_factory=list,
        description="important info the data has that the answer omitted",
    )


DECOMPOSE_SUFFIX = (
    "\n\nBreak the question above into one or more focused sub-questions "
    "suitable for retrieving supporting passages. If the question is already "
    "focused, return it unchanged as the only entry."
)

ANSWER_SUFFIX = (
    "\n\nUsing only the context passages above, list the atomic facts that "
    "support an answer, each citing the id of the single chunk it came from. "
    "Then write the final answer from those facts alone - the answer must "
    "not say anything its facts don't already say. If the context does not "
    "support an answer, say so plainly and leave facts empty."
)

VERIFICATION_SUFFIX = (
    "\n\nFor each fact in the answer above, check it against the chunk it "
    "cited: mark it 'yes' if the chunk fully supports it, 'partial' if only "
    "partly, 'no' if unsupported, and quote the verbatim span backing it (or "
    "quote empty if unsupported). List separately any claims in the answer "
    "text that contradict the retrieved data."
)

COMPLETENESS_SUFFIX = (
    "\n\nCompare the answer above against all the retrieved context. Is it "
    "complete? List any important information the context contains that the "
    "answer omitted."
)


class Decomposer(Role[DecomposeOutput]):
    NAME = "decompose"
    SUFFIX = DECOMPOSE_SUFFIX
    Output = DecomposeOutput


class Answerer(Role[AnswerOutput]):
    NAME = "answer"
    SUFFIX = ANSWER_SUFFIX
    Output = AnswerOutput


class Verifier(Role[VerificationOutput]):
    NAME = "verify"
    SUFFIX = VERIFICATION_SUFFIX
    Output = VerificationOutput


class CompletenessChecker(Role[CompletenessOutput]):
    NAME = "completeness"
    SUFFIX = COMPLETENESS_SUFFIX
    Output = CompletenessOutput


def format_query_prefix(question: str) -> str:
    return f"Question: {json.dumps(question)}"


def format_prompt_prefix(question: str, chunks: list[SearchResult]) -> str:
    return f"Context:\n{_format_context(chunks)}\n\nQuestion: {json.dumps(question)}"


def format_answer_suffix(answer: AnswerOutput) -> str:
    return f"\n\nAnswer:\n{answer.model_dump_json()}"
