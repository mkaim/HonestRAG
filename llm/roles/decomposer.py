from pydantic import BaseModel, Field

from llm.agent import Role

DECOMPOSE_SUFFIX = (
    "\n\nBreak the question above into one or more focused sub-questions "
    "suitable for retrieving supporting passages. Give each distinct piece "
    "of information the question asks for its own sub-question, so each can "
    "be answered or declined on its own. Each sub-question must make sense "
    "alone: name its subject instead of using pronouns. If the question is "
    "already focused, return it unchanged as the only entry. Keep every specific "
    "claim the question makes - dates, numbers, names, stated facts - in the "
    "sub-questions, even ones that look wrong: do not correct or drop them, "
    "since later steps check them against the sources."
)


class DecomposeOutput(BaseModel):
    questions: list[str] = Field(
        description="one or more focused sub-questions to retrieve context for"
    )


class Decomposer(Role[DecomposeOutput]):
    NAME = "decompose"
    SUFFIX = DECOMPOSE_SUFFIX
    Output = DecomposeOutput
