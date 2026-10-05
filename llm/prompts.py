import json

from models import SearchResult


def _format_context(chunks: list[SearchResult]) -> str:
    return "\n\n".join(f"[{r.chunk.id}] {r.chunk.content}" for r in chunks)


def format_query_prefix(question: str) -> str:
    return f"Question: {json.dumps(question)}"


def format_prompt_prefix(
    question: str, chunks: list[SearchResult], questions: list[str]
) -> str:
    return (
        f"Context:\n{_format_context(chunks)}\n\n"
        f"Question: {json.dumps(question)}"
        f"{format_subquestions_suffix(questions)}"
    )


def format_subquestions_suffix(questions: list[str]) -> str:
    return f"\n\nSub-questions:\n{json.dumps(questions)}"


def chunks_by_id(chunks: list[SearchResult]) -> dict[str, str]:
    return {r.chunk.id: r.chunk.content for r in chunks}
