import json
from types import SimpleNamespace

import pytest

import server
from llm.roles.answerer import AnswerOutput, Fact, SubAnswer
from llm.roles.completeness import CompletenessOutput, SubQuestionCoverage
from llm.roles.decomposer import DecomposeOutput
from llm.roles.verifier import FactCheck, VerificationOutput
from models import Chunk, SearchResult

CHUNK = Chunk(document_id="doc", index=0, content="OAPEC proclaimed an oil embargo.")
QUOTE = "OAPEC proclaimed an oil embargo"

OUTPUTS = {
    DecomposeOutput: DecomposeOutput(questions=["Who?", "Warsaw?"]),
    AnswerOutput: AnswerOutput(
        subanswers=[
            SubAnswer(
                question="Who?",
                facts=[Fact(chunk_id=CHUNK.id, statement="OAPEC did.", quote=QUOTE)],
                answer="OAPEC.",
            ),
            SubAnswer(question="Warsaw?", facts=[], answer="Not in the sources."),
        ],
        answer="OAPEC; Warsaw isn't covered.",
    ),
    VerificationOutput: VerificationOutput(
        facts=[
            FactCheck(
                statement="OAPEC did.", supported="yes", chunk_id=CHUNK.id, quote=QUOTE
            )
        ]
    ),
    CompletenessOutput: CompletenessOutput(
        subquestions=[
            SubQuestionCoverage(
                question="Who?", status="covered", chunk_id=CHUNK.id, quote=QUOTE
            ),
            SubQuestionCoverage(question="Warsaw?", status="not_in_sources"),
        ]
    ),
}


class FakeAgent:
    async def run(self, prompt: str, output_type):
        return SimpleNamespace(
            output=OUTPUTS[output_type], new_messages_json=lambda: b"[]"
        )


class FakeRag:
    def __init__(self, results: list[SearchResult]):
        self.results = results

    async def search(self, queries: list[str]) -> list[list[SearchResult]]:
        return [self.results for _ in queries]


async def _events(monkeypatch, results: list[SearchResult]) -> list[dict]:
    monkeypatch.setattr(server, "agent", FakeAgent())
    monkeypatch.setattr(server, "rag", FakeRag(results))
    return [
        json.loads(line.removeprefix("data: "))
        async for line in server._ask_stream("question")
    ]


@pytest.mark.parametrize("debug", [False, True])
async def test_ask_stream_runs_every_stage(monkeypatch, debug):
    monkeypatch.setattr(server.cfg, "debug", debug)

    events = await _events(monkeypatch, [SearchResult(chunk=CHUNK)])

    assert [(e["stage"], e["status"]) for e in events] == [
        ("decompose", "running"),
        ("decompose", "done"),
        ("search", "running"),
        ("search", "done"),
        ("answer", "running"),
        ("answer", "done"),
        ("verify", "running"),
        ("verify", "done"),
        ("completeness", "running"),
        ("completeness", "done"),
        ("final", "done"),
    ]
    stage_results = [
        e for e in events if e["status"] == "done" and e["stage"] != "final"
    ]
    assert all(("debug" in e) == debug for e in stage_results)
    final = events[-1]["data"]
    assert final["answer"] == "OAPEC; Warsaw isn't covered."
    assert final["questions"] == ["Who?", "Warsaw?"]
    assert final["subanswers"][0]["facts"][0] == {
        "chunk_id": CHUNK.id,
        "statement": "OAPEC did.",
        "quote": QUOTE,
        "verified": True,
    }
    assert final["stats"] == {
        "total_facts": 1,
        "answer_quotes_verified": 1,
        "verifier_fully_supported": 1,
        "verifier_partially_supported": 0,
        "verifier_unsupported": 0,
        "verifier_hallucinated_quotes": 0,
        "verifier_no_claims": 0,
        "subquestions_covered": 1,
        "subquestions_not_in_sources": 1,
        "subquestions_missed": 0,
        "subquestions_total": 2,
        "complete": True,
    }


async def test_ask_stream_without_passages_declines(monkeypatch):
    events = await _events(monkeypatch, [])

    assert [e["stage"] for e in events] == [
        "decompose",
        "decompose",
        "search",
        "search",
        "final",
    ]
    final = events[-1]["data"]
    assert final["answer"] == "No matching passages found in the corpus."
    assert final["subanswers"] == []
    assert final["completeness"]["not_in_sources"] == ["Who?", "Warsaw?"]
    assert final["stats"]["total_facts"] == 0
    assert final["stats"]["subquestions_not_in_sources"] == 2
    assert final["stats"]["complete"] is True
