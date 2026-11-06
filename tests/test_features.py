"""Tests for the v2 features: BM25, hybrid search, re-ranking, calibrated scoring, uploads, API."""
import numpy as np
import pytest

from qa import config
from qa.bm25 import BM25, tokenize
from qa.documents import UploadError, chunk_text, extract_text

needs_index = pytest.mark.skipif(not (config.INDEX_DIR / "passages.faiss").exists(),
                                 reason="index not built; run `python -m qa.build_index`")


# ---------- pure unit tests (no models) ----------

def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("Who won the Super Bowl 50?") == ["won", "super", "bowl", "50"]


def test_bm25_ranks_lexical_match_first():
    docs = ["The cat sat on the mat.", "Denver Broncos won Super Bowl 50.", "Paris is the capital of France."]
    bm = BM25(docs)
    top = bm.search("Which team won Super Bowl 50?", k=3)
    assert top[0][0] == 1
    assert all(s > 0 for _, s in top)
    assert bm.search("zebra", k=3) == []


def test_chunk_text_overlaps_and_covers_text():
    text = " ".join(f"Sentence number {i} is here." for i in range(100))
    chunks = chunk_text(text, words=50, overlap=10)
    assert len(chunks) > 5
    assert all(len(c.split()) <= 100 for c in chunks)
    assert "Sentence number 0 " in chunks[0] and "Sentence number 99 " in chunks[-1] + " "
    # consecutive chunks share text (overlap)
    assert set(chunks[0].split()[-5:]) & set(chunks[1].split()[:15])


def _tiny_pdf(text: str) -> bytes:
    """A minimal single-page PDF with one line of text (no PDF library needed)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


def test_extract_text_from_pdf_and_txt():
    assert "Krishna Siva Prasad" in extract_text("r.pdf", _tiny_pdf("The guide was Dr. M. Krishna Siva Prasad."))
    assert extract_text("a.txt", b"Hello world, this is plain text.").startswith("Hello")
    with pytest.raises(UploadError):
        extract_text("a.docx", b"whatever")
    with pytest.raises(UploadError):
        extract_text("empty.txt", b"  ")


# ---------- integration tests (models + index) ----------

@pytest.fixture(scope="module")
def pipe():
    from qa.pipeline import QAPipeline
    return QAPipeline()


@needs_index
@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
@pytest.mark.parametrize("rerank", [False, True])
def test_search_modes_find_gold_passage(pipe, mode, rerank):
    hits = pipe.engine.search("Who was the Super Bowl 50 MVP?", k=3, mode=mode, rerank=rerank)
    assert len(hits) == 3
    assert any("Super Bowl 50" in h.passage.title for h in hits)
    assert abs(sum(h.prior for h in hits) - 1) < 1e-4
    assert [h.score for h in hits] == sorted([h.score for h in hits], reverse=True)


@needs_index
@pytest.mark.parametrize("scoring", ["raw", "calibrated"])
def test_extractive_scoring_modes(pipe, scoring):
    res = pipe.ask("Which network broadcasted Super Bowl 50 in the United States?", scoring=scoring)
    assert res.answer == "CBS"
    assert 0 < res.confidence <= 1
    assert res.alternatives and res.alternatives[0]["text"] == res.answer
    src = res.retrieved[res.passage_index]["text"]
    assert src[res.char_start:res.char_start + len(res.answer)] == res.answer
    assert set(res.timings) >= {"retrieve_ms", "rerank_ms", "read_ms"}


@needs_index
def test_compare_reports_agreement(pipe):
    out = pipe.compare("Which network broadcasted Super Bowl 50 in the United States?")
    assert out["extractive"]["answer"] == "CBS" and out["generative"]["answer"] == "CBS"
    assert out["agreement"]["agree"] is True
    assert 0 < out["generative"]["confidence"] <= 1


@needs_index
def test_uploaded_documents_are_searchable(pipe):
    pipe.documents.clear()
    with pytest.raises(LookupError):
        pipe.ask("anything?", corpus="docs")
    doc = pipe.documents.add("notes.txt", b"The internship ran from 1 June 2025 to 1 August 2025. "
                                          b"The fine-tuning run took 37 minutes on a laptop CPU.")
    try:
        res = pipe.ask("How long did the fine-tuning run take?", corpus="docs")
        assert "37 minutes" in res.answer
        assert res.source_title.startswith("notes.txt")
    finally:
        pipe.documents.remove(doc.doc_id)
    assert pipe.documents.engine is None


@needs_index
def test_api_v2(pipe):
    from fastapi.testclient import TestClient
    from qa.api import app
    with TestClient(app) as client:
        h = client.get("/api/health").json()
        assert h["status"] == "ok" and h["passages"] == 2067
        r = client.post("/api/ask", json={"question": "Which network broadcasted Super Bowl 50 in the United States?",
                                          "gold": ["CBS"]}).json()
        assert r["mode"] == "single" and r["gold_score"]["em"] == 1.0
        c = client.post("/api/ask", json={"question": "Who was the Super Bowl 50 MVP?", "reader": "compare"}).json()
        assert c["mode"] == "compare" and "agreement" in c
        assert client.post("/api/ask", json={"question": "Who?", "corpus": "docs"}).status_code == 400
        up = client.post("/api/documents", files={"file": ("n.txt", b"QA Studio was built in November 2026 by the team.")})
        assert up.status_code == 200
        doc_id = up.json()["document"]["doc_id"]
        a = client.post("/api/ask", json={"question": "When was QA Studio built?", "corpus": "docs"}).json()
        assert "November 2026" in a["answer"]
        assert client.delete(f"/api/documents/{doc_id}").json()["documents"] == []
        assert client.post("/api/documents", files={"file": ("x.exe", b"MZ")}).status_code == 400
        rnd = client.get("/api/random").json()
        assert rnd["question"] and rnd["gold"]
        assert client.get("/api/results").json()["baseline"] is not None
        assert "QA Studio" in client.get("/").text
        assert client.get("/static/app.js").status_code == 200
