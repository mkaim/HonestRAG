import pytest

from fusion import RRF
from models import Chunk, Document, SearchMode
from rag import Rag

CHUNKS = [
    Chunk(
        document_id="d1",
        index=0,
        content="The cat sat on the warm windowsill in the afternoon sun.",
    ),
    Chunk(
        document_id="d2",
        index=0,
        content="Postgres full text search uses BM25 ranking via ParadeDB.",
    ),
    Chunk(
        document_id="d3",
        index=0,
        content="Vector embeddings capture semantic similarity between sentences.",
    ),
    Chunk(
        document_id="d4",
        index=0,
        content="A dog chased the ball across the green park.",
    ),
    Chunk(
        document_id="d5",
        index=0,
        content="HNSW indexes speed up approximate nearest neighbour search.",
    ),
]


async def _add_all(db) -> None:
    for chunk in CHUNKS:
        await db.add(Document(id=chunk.document_id), [chunk])


@pytest.fixture
async def populated_db(rag_db):
    await _add_all(rag_db)
    return rag_db


async def test_add_is_idempotent(rag_db):
    await _add_all(rag_db)
    await _add_all(rag_db)  # upsert, no duplicates / no error

    async with rag_db.pool.connection() as conn:
        cur = await conn.execute("SELECT count(*) FROM chunk")
        (count,) = await cur.fetchone()
    assert count == len(CHUNKS)


async def test_bm25_search_ranks_lexical_match_first(populated_db):
    [results] = await populated_db.search(SearchMode.BM25, ["BM25 ranking ParadeDB"], 5)

    assert results, "expected at least one BM25 hit"
    assert results[0].chunk.content == CHUNKS[1].content
    assert SearchMode.BM25.value in results[0].scores


async def test_semantic_search_ranks_by_meaning(populated_db):
    [results] = await populated_db.search(
        SearchMode.SEMANTIC, ["finding text with similar meaning"], 5
    )

    top_contents = [r.chunk.content for r in results[:2]]
    assert CHUNKS[2].content in top_contents  # "semantic similarity between sentences"


async def test_unknown_mode_raises(rag_db):
    with pytest.raises(ValueError):
        await rag_db.search("KEYWORD", ["x"], 5)  # type: ignore[arg-type]


async def test_rag_fuses_both_modes(populated_db):
    rag = Rag(populated_db, RRF(), retrieval_limit=5, rf_limit=5)

    [ranked] = await rag.search(["nearest neighbour vector search"])

    assert ranked
    assert all(RRF.SCORE_FIELD in r.scores for r in ranked)
    scores = [r.scores[RRF.SCORE_FIELD] for r in ranked]
    assert scores == sorted(scores, reverse=True)


async def test_rag_handles_multiple_queries(populated_db):
    rag = Rag(populated_db, RRF(), retrieval_limit=5, rf_limit=5)

    results = await rag.search(["a cat in the sun", "approximate nearest neighbour"])

    assert len(results) == 2
    assert all(len(r) > 0 for r in results)


def test_rrf_merge_pure():
    from models import SearchResult

    a = Chunk(document_id="a", index=0, content="a")
    b = Chunk(document_id="b", index=0, content="b")

    ranking_1 = [
        SearchResult(chunk=a).with_score("x", 1.0),
        SearchResult(chunk=b).with_score("x", 0.5),
    ]
    ranking_2 = [
        SearchResult(chunk=b).with_score("y", 2.0),
        SearchResult(chunk=a).with_score("y", 1.0),
    ]

    merged = RRF(k=1).merge([ranking_1, ranking_2], limit=10)

    assert {r.chunk.id for r in merged} == {"a#0", "b#0"}
    # both chunks appear at ranks 1 and 2 across the two rankings -> equal RRF score
    for r in merged:
        assert r.scores[RRF.SCORE_FIELD] == pytest.approx(1 / 2 + 1 / 3)
        assert "x" in r.scores and "y" in r.scores  # per-mode scores preserved

    # inputs untouched
    assert RRF.SCORE_FIELD not in ranking_1[0].scores
    assert ranking_2[0].scores == {"y": 2.0}
