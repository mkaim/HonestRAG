import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import server
from models import Chunk, Document

_FIXTURES = Path(__file__).parent / "fixtures"


class FakeDb:
    def __init__(self):
        self.added: list[tuple[Document, list[Chunk]]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def add(self, document: Document, chunks: list[Chunk]) -> None:
        self.added.append((document, chunks))


@pytest.fixture
def client(monkeypatch):
    fake_db = FakeDb()
    monkeypatch.setattr(server, "db", fake_db)
    with TestClient(server.app) as c:
        c.fake_db = fake_db
        yield c


def _events(response) -> list[dict]:
    """Parses a text/event-stream response body into its data: {...}
    events, same convention as /api/ask's own SSE stream."""
    return [
        json.loads(line.removeprefix("data: "))
        for block in response.text.split("\n\n")
        if block
        for line in block.split("\n")
        if line.startswith("data: ")
    ]


def test_upload_single_file_uses_filename_as_id(client):
    response = client.post(
        "/api/documents",
        files={"file": ("notes.txt", b"first\n\nsecond", "text/plain")},
    )

    assert response.status_code == 200
    events = _events(response)
    assert [(e["stage"], e["status"]) for e in events] == [
        ("notes.txt", "running"),
        ("notes.txt", "done"),
        ("final", "done"),
    ]
    assert events[-1]["data"] == {
        "documents": ["notes.txt"],
        "failed": [],
        "chunks": 1,
    }
    [(document, chunks)] = client.fake_db.added
    assert document.id == "notes.txt"
    assert len(chunks) == 1


def test_upload_single_file_with_explicit_id(client):
    response = client.post(
        "/api/documents",
        params={"id": "custom-id"},
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 200
    events = _events(response)
    assert events[-1]["data"]["documents"] == ["custom-id"]
    [(document, _)] = client.fake_db.added
    assert document.id == "custom-id"


def test_upload_pdf_fixture(client):
    response = client.post(
        "/api/documents",
        files={
            "file": (
                "sample.pdf",
                (_FIXTURES / "sample.pdf").read_bytes(),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200
    final = _events(response)[-1]["data"]
    assert final["documents"] == ["sample.pdf"]
    assert final["chunks"] >= 1


def test_upload_unsupported_extension_is_reported_as_failed(client):
    response = client.post(
        "/api/documents",
        files={"file": ("data.csv", b"a,b\n1,2", "text/csv")},
    )

    assert response.status_code == 200
    events = _events(response)
    assert [(e["stage"], e["status"]) for e in events] == [
        ("data.csv", "running"),
        ("data.csv", "error"),
        ("final", "done"),
    ]
    assert events[-1]["data"] == {"documents": [], "failed": ["data.csv"], "chunks": 0}
    assert client.fake_db.added == []


def test_upload_zip_ingests_each_file_by_name(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("a.txt", "first\n\nsecond")
        archive.writestr("b.txt", "third")
    buf.seek(0)

    response = client.post(
        "/api/documents",
        files={"file": ("docs.zip", buf.read(), "application/zip")},
    )

    assert response.status_code == 200
    final = _events(response)[-1]["data"]
    assert sorted(final["documents"]) == ["a.txt", "b.txt"]
    assert final["chunks"] == 2
    ids = sorted(document.id for document, _ in client.fake_db.added)
    assert ids == ["a.txt", "b.txt"]


def test_upload_zip_skips_unsupported_files_like_dir_ingest(client):
    """Same convention as ingest.py's --dir: a mixed-format zip ingests
    what it can and silently skips the rest, rather than failing the
    whole upload over one unparseable file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("good.txt", "content")
        archive.writestr("image.png", b"not really a png")
    buf.seek(0)

    response = client.post(
        "/api/documents",
        files={"file": ("docs.zip", buf.read(), "application/zip")},
    )

    assert response.status_code == 200
    final = _events(response)[-1]["data"]
    assert final["documents"] == ["good.txt"]
    assert final["failed"] == []
    assert final["chunks"] == 1
    [(document, _)] = client.fake_db.added
    assert document.id == "good.txt"


def test_upload_zip_rejects_explicit_id(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("a.txt", "content")
    buf.seek(0)

    response = client.post(
        "/api/documents",
        params={"id": "custom-id"},
        files={"file": ("docs.zip", buf.read(), "application/zip")},
    )

    assert response.status_code == 400
    assert client.fake_db.added == []


def test_upload_zip_with_no_supported_files_is_rejected(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("image.png", b"not really a png")
    buf.seek(0)

    response = client.post(
        "/api/documents",
        files={"file": ("docs.zip", buf.read(), "application/zip")},
    )

    assert response.status_code == 400
    assert client.fake_db.added == []
