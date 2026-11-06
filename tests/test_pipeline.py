"""Integration tests: need the downloaded models and a built index (`python -m qa.build_index`)."""
import pytest

from qa import config

pytestmark = pytest.mark.skipif(not (config.INDEX_DIR / "passages.faiss").exists(),
                                reason="index not built; run `python -m qa.build_index`")

CONTEXT = "France, officially known as the French Republic, has its capital in Paris."


@pytest.fixture(scope="module")
def pipe():
    from qa.pipeline import QAPipeline
    return QAPipeline()


def test_extractive_reader_reads_given_context():
    from qa.readers import ExtractiveReader
    ans = ExtractiveReader().read("What is the capital of France?", [CONTEXT])
    assert ans.text == "Paris"
    assert ans.passage_index == 0


def test_extractive_reader_handles_long_context():
    from qa.readers import ExtractiveReader
    filler = "This sentence is filler text about nothing in particular. " * 80
    ans = ExtractiveReader().read("What is the capital of France?", [filler + CONTEXT])
    assert ans.text == "Paris"


def test_retriever_finds_super_bowl_passage(pipe):
    hits = pipe.retriever.search(["Who was the Super Bowl 50 MVP?"], k=3)[0]
    assert any("Super Bowl" in p.title for p, _ in hits)


def test_end_to_end_extractive(pipe):
    res = pipe.ask("Which network broadcasted Super Bowl 50 in the United States?", reader="extractive", top_k=3)
    assert res.answer == "CBS"


def test_end_to_end_generative(pipe):
    res = pipe.ask("Who was the Super Bowl 50 MVP?", reader="generative", top_k=3)
    assert "Von Miller" in res.answer


def test_api(pipe):
    from fastapi.testclient import TestClient
    from qa.api import app
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/ask", json={"question": "Which network broadcasted Super Bowl 50 in the United States?"})
        assert r.status_code == 200
        assert r.json()["answer"] == "CBS"
        assert client.post("/ask", json={"question": "x"}).status_code == 422
        assert "<form" in client.get("/").text  # web app
