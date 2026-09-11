from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from models import SearchResult

SYSTEM_PROMPT = (
    "You answer the user's question using only the numbered context passages "
    "provided. Cite the passages you rely on as [1], [2], etc. If the context "
    "does not contain the answer, say so plainly rather than guessing."
)


@dataclass
class Answer:
    text: str
    passages: list[SearchResult]


def build_agent(base_url: str, api_key: str, model: str) -> Agent:
    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    return Agent(
        OpenAIChatModel(model, provider=provider),
        system_prompt=SYSTEM_PROMPT,
    )


def _format_context(passages: list[SearchResult]) -> str:
    return "\n\n".join(
        f"[{i}] {p.chunk.content}" for i, p in enumerate(passages, start=1)
    )


async def answer(agent: Agent, question: str, passages: list[SearchResult]) -> Answer:
    prompt = f"Context:\n{_format_context(passages)}\n\nQuestion: {question}"
    result = await agent.run(prompt)
    return Answer(text=result.output, passages=passages)
