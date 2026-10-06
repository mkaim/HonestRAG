from llm.roles.completeness import (
    CompletenessChecker,
    CompletenessOutput,
    SubQuestionCoverage,
)

CONTENTS = {"doc#0": "OAPEC proclaimed an oil embargo in October 1973."}


def _verify(*subquestions: SubQuestionCoverage):
    output = CompletenessOutput(subquestions=list(subquestions))
    return CompletenessChecker(agent=None).verify(output, CONTENTS)


def test_verified_quote_is_covered():
    result = _verify(
        SubQuestionCoverage(
            question="Who?",
            status="covered",
            chunk_id="doc#0",
            quote="OAPEC proclaimed an oil embargo",
        )
    )

    assert result.covered == ["Who?"]
    assert result.complete
    assert result.hallucinated_quotes == 0


def test_covered_claim_with_unverified_quote_is_missed():
    result = _verify(
        SubQuestionCoverage(
            question="Who?", status="covered", chunk_id="doc#0", quote="OPEC did it"
        )
    )

    assert result.missed == ["Who?"]
    assert not result.complete
    assert result.hallucinated_quotes == 1
    assert "downgraded: quote not found in doc#0" in result.subquestions[0].reason


def test_not_in_sources_keeps_answer_complete():
    result = _verify(
        SubQuestionCoverage(
            question="Who?",
            status="covered",
            chunk_id="doc#0",
            quote="OAPEC proclaimed an oil embargo",
        ),
        SubQuestionCoverage(question="Warsaw fuel prices?", status="not_in_sources"),
    )

    assert result.not_in_sources == ["Warsaw fuel prices?"]
    assert result.complete
    assert result.to_dict()["missed"] == []


def test_missed_makes_answer_incomplete():
    result = _verify(SubQuestionCoverage(question="When?", status="missed"))

    assert result.missed == ["When?"]
    assert not result.complete
