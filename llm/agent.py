import re
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

SYSTEM_PROMPT = (
    "You answer questions using only the retrieved context passages you are "
    "given. Never use outside knowledge. If the context does not support an "
    "answer, say so plainly instead of guessing."
)


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


def _normalize(text: str) -> str:
    text = re.sub(r"[\"'“”‘’]", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


# "...", "…", optionally bracketed as "[...]" or "(…)".
_ELLIPSIS = re.compile(r"[\[(]?(?:\.{3,}|…)[\])]?")
# Each part of an elided quote must be this long, so a quote can't pass by
# stitching together fragments short enough to match almost anywhere.
_MIN_SEGMENT_WORDS = 3


def quote_in_chunk(quote: str, chunk_content: str) -> bool:
    """Deterministic check that `quote` actually appears in `chunk_content`,
    tolerant of whitespace and quote-mark differences. A quote may skip text
    with an ellipsis; each part must then appear verbatim, in order. This is
    the ground truth for whether an LLM-produced quote is real or
    hallucinated - it does not trust the LLM's own supported/quote claim."""
    chunk = _normalize(chunk_content)
    whole = _normalize(quote)
    if not whole:
        return False
    if whole in chunk:
        return True

    segments = [s for s in map(_normalize, _ELLIPSIS.split(quote)) if s]
    if not segments or segments == [whole]:
        return False  # nothing but ellipses, or no ellipsis and no match
    position = 0
    for segment in segments:
        if len(segment.split()) < _MIN_SEGMENT_WORDS:
            return False
        found = chunk.find(segment, position)
        if found < 0:
            return False
        position = found + len(segment)
    return True


def _chunk_content(chunk_id: str, contents: dict[str, str]) -> str:
    """Looks up chunk_id in contents, tolerant of the model echoing it back
    wrapped in the "[chunk_id]" brackets used in the context we show it."""
    return contents.get(chunk_id, contents.get(chunk_id.strip("[]"), ""))


@dataclass(frozen=True)
class QuoteResult:
    """One quote's deterministic verification result: whether it was found,
    verbatim, in the chunk it claims to come from."""

    chunk_id: str
    quote: str
    verified: bool
