"""FastAPI service + web front end.

    uvicorn qa.api:app --port 8000
    open http://127.0.0.1:8000        (web app)
         http://127.0.0.1:8000/docs   (Swagger UI)

JSON API (all under /api):
    POST   /api/ask              question -> answer(s), evidence, confidence, timings
    POST   /api/documents        upload a PDF / TXT / MD file (multipart field "file")
    GET    /api/documents        list uploaded documents
    DELETE /api/documents/{id}   remove one
    GET    /api/random           a random SQuAD validation question with its gold answers
    GET    /api/results          measured results (baseline, advanced, training)
    GET    /api/health           status + model names
`POST /ask` and `GET /health` are kept for backward compatibility.
"""
import json
import random
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .documents import UploadError
from .metrics import exact_match, f1_score, max_over_truths
from .pipeline import QAPipeline

WEB_DIR = Path(__file__).parent / "web"
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipe = QAPipeline()
    pipe.reader("extractive")  # warm the default reader; Flan-T5 and the re-ranker load on first use
    state["pipe"] = pipe
    yield
    state.clear()


app = FastAPI(title="Question Answering in LLMs", version="2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def pipe() -> QAPipeline:
    p = state.get("pipe")
    if p is None:
        raise HTTPException(503, "Models are still loading")
    return p


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, examples=["Who was the Super Bowl 50 MVP?"])
    reader: Literal["extractive", "generative", "compare"] = "extractive"
    top_k: int = Field(3, ge=1, le=10)
    retrieval: Literal["dense", "bm25", "hybrid"] = config.DEFAULT_RETRIEVAL
    rerank: bool = config.DEFAULT_RERANK
    scoring: Literal["raw", "calibrated"] = config.DEFAULT_SCORING
    corpus: Literal["squad", "docs"] = "squad"
    gold: List[str] | None = Field(None, description="optional reference answers; the response is scored against them")


def _score(answer: str, gold: List[str] | None):
    if not gold:
        return None
    return {"em": max_over_truths(exact_match, answer, gold), "f1": round(max_over_truths(f1_score, answer, gold), 3)}


@app.post("/api/ask")
def api_ask(req: AskRequest):
    p = pipe()
    kw = dict(top_k=req.top_k, retrieval=req.retrieval, rerank=req.rerank, scoring=req.scoring, corpus=req.corpus)
    try:
        if req.reader == "compare":
            out = p.compare(req.question, **kw)
            for k in ("extractive", "generative"):
                out[k]["gold_score"] = _score(out[k]["answer"], req.gold)
            out["mode"] = "compare"
        else:
            out = p.ask(req.question, reader=req.reader, **kw).to_dict()
            out["gold_score"] = _score(out["answer"], req.gold)
            out["mode"] = "single"
    except LookupError as exc:
        raise HTTPException(400, str(exc))
    out["gold"] = req.gold
    return out


@app.post("/api/documents")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    try:
        doc = pipe().documents.add(file.filename or "upload.txt", data)
    except UploadError as exc:
        raise HTTPException(400, str(exc))
    return {"document": doc.__dict__, "documents": pipe().documents.list()}


@app.get("/api/documents")
def list_documents():
    return {"documents": pipe().documents.list()}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    if not pipe().documents.remove(doc_id):
        raise HTTPException(404, "No such document")
    return {"documents": pipe().documents.list()}


_squad_cache = {}


@app.get("/api/random")
def random_question():
    if "rows" not in _squad_cache:
        from .data import load_squad
        ds = load_squad("validation")
        _squad_cache["rows"] = list(zip(ds["question"], ds["answers"], ds["title"]))
    q, a, t = random.choice(_squad_cache["rows"])
    return {"question": q.strip(), "gold": sorted(set(a["text"]), key=len), "title": t.replace("_", " ")}


@app.get("/api/results")
def results():
    out = {}
    for key, name in [("baseline", "results.json"), ("advanced", "advanced.json"), ("training", "training_log.json")]:
        path = config.RESULTS_DIR / name
        out[key] = json.loads(path.read_text()) if path.exists() else None
    return out


@app.get("/api/health")
def api_health():
    p = state.get("pipe")
    return {
        "status": "ok" if p else "loading",
        "passages": len(p.engine) if p else 0,
        "documents": len(p.documents.docs) if p else 0,
        "extractive_model": Path(config.extractive_reader_path()).name,
        "generative_model": config.GEN_MODEL,
        "embed_model": config.EMBED_MODEL,
        "rerank_model": config.RERANK_MODEL,
        "defaults": {"retrieval": config.DEFAULT_RETRIEVAL, "rerank": config.DEFAULT_RERANK,
                     "scoring": config.DEFAULT_SCORING},
    }


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html")


# ---- v1 endpoints (kept for scripts and tests written against the first version) ----

class LegacyAsk(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    reader: Literal["extractive", "generative"] = "extractive"
    top_k: int = Field(3, ge=1, le=10)


@app.post("/ask", include_in_schema=False)
def ask(req: LegacyAsk):
    data = pipe().ask(req.question, reader=req.reader, top_k=req.top_k).to_dict()
    data.pop("source_passage")
    return data


@app.get("/health", include_in_schema=False)
def health():
    return api_health()
