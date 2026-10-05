from pydantic import BaseModel, Field

from llm.agent import Role

DECOMPOSE_SUFFIX = (
    "\n\nBreak the question above into one or more focused sub-questions "
    "suitable for retrieving supporting passages. If the question is already "
    "focused, return it unchanged as the only entry."
)


class DecomposeOutput(BaseModel):
    questions: list[str] = Field(
        description="one or more focused sub-questions to retrieve context for"
    )


class Decomposer(Role[DecomposeOutput]):
    NAME = "decompose"
    SUFFIX = DECOMPOSE_SUFFIX
    Output = DecomposeOutput
