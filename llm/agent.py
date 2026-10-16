import re
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

SYSTEM_PROMPT = (
    "You answer questions using only the retrieved context passages you are "
    "given. Never use outside knowledge. If the context does not support an "
    "answer, say so plainly instead of guessing."
)


def build_agent(base_url: str, api_key: str, model: str) -> Agent:
    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    return Agent(
        OpenAIChatModel(model, provider=provider), system_prompt=SYSTEM_PROMPT
    )


def describe_output_fields(model: type[BaseModel]) -> str:
    """A JSON skeleton for `model` with a `// comment` per field, in place
    of pydantic-ai's default schema instructions (raw JSON Schema, with
    $ref/$defs indirection for nested models) - meant to be easier for a
    model to read than JSON Schema, especially a smaller/local one, while
    staying closer to the JSON shape it must actually produce than a prose
    field list would. Only handles the shapes this project's own Output
    models actually use: flat string/array fields, string enums, and one
    level of nesting via $ref - not a general JSON Schema renderer."""
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})
    skeleton = "\n".join(_render_object(schema, defs, indent=""))
    return (
        "Respond with a single JSON object matching this structure - no "
        f"other text, no Markdown fencing:\n{skeleton}"
    )


def _render_object(obj_schema: dict, defs: dict, indent: str) -> list[str]:
    """Lines for a JSON object, e.g. ['{', '  "a": "...",', '  "b": "..."', '}'].
    Every returned line already carries its own full indent (including the
    first and last) - callers that place this inline after a key (as
    `_render_value` does) must strip the leading indent of line 0
    themselves, since that line continues the `"key": ` on its own line."""
    required = set(obj_schema.get("required", []))
    inner = indent + "  "
    lines = [f"{indent}{{"]
    props = list(obj_schema.get("properties", {}).items())
    for i, (name, prop) in enumerate(props):
        is_last = i == len(props) - 1
        value_lines = _render_value(prop, defs, inner)
        value_lines[0] = f'{inner}"{name}": {value_lines[0]}'
        value_lines[-1] += "" if is_last else ","
        value_lines[-1] += _trailing_comment(prop, optional=name not in required)
        lines.extend(value_lines)
    lines.append(f"{indent}}}")
    return lines


def _render_value(prop: dict, defs: dict, indent: str) -> list[str]:
    """Lines for one field's value, meant to be placed right after `"key": `
    - line 0 carries no indent of its own (it continues that line); every
    other line is already indented at `indent` or deeper."""
    if "$ref" in prop:
        lines = _render_object(defs[prop["$ref"].rsplit("/", 1)[-1]], defs, indent)
        return [lines[0].removeprefix(indent), *lines[1:]]
    if prop.get("type") == "array":
        item_lines = _render_value(prop["items"], defs, indent + "  ")
        return [
            "[",
            f"{indent}  {item_lines[0]}",
            *item_lines[1:],
            f"{indent}]",
        ]
    if "enum" in prop:
        return [" | ".join(f'"{v}"' for v in prop["enum"])]
    return ['"..."']


def _trailing_comment(prop: dict, *, optional: bool) -> str:
    parts = [prop["description"]] if prop.get("description") else []
    if optional:
        parts.append("optional")
    return f"  // {' - '.join(parts)}" if parts else ""


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

    def __init__(self, agent: Agent, simplified_schema: bool = False):
        self.agent = agent
        # Local providers don't include the output schema in what the
        # model reads, so simplified_schema appends our own JSON-skeleton
        # field description (describe_output_fields) to the prompt instead.
        self.simplified_schema = simplified_schema

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
        if self.simplified_schema:
            prompt += f"\n\n{describe_output_fields(self.Output)}"
            output_type = NativeOutput(self.Output, template=False)
        else:
            output_type = NativeOutput(self.Output)
        result = await self.agent.run(prompt, output_type=output_type)
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
