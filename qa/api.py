"""FastAPI demo.

    uvicorn qa.api:app --port 8000
    open http://127.0.0.1:8000  (web page)  or  http://127.0.0.1:8000/docs  (Swagger UI)
"""
from contextlib import asynccontextmanager
from typing import List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import config
from .pipeline import QAPipeline

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipe = QAPipeline()
    pipe.reader("extractive")  # warm the default reader; Flan-T5 loads on first use
    state["pipe"] = pipe
    yield
    state.clear()


app = FastAPI(title="Question Answering in LLMs", version="1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, examples=["Who was the Super Bowl 50 MVP?"])
    reader: Literal["extractive", "generative"] = "extractive"
    top_k: int = Field(3, ge=1, le=10)


class Passage(BaseModel):
    pid: int
    title: str
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    reader: str
    score: float | None
    source_title: str | None
    retrieved: List[Passage]
    latency_ms: float


@app.get("/health")
def health():
    pipe = state.get("pipe")
    return {
        "status": "ok" if pipe else "loading",
        "passages": len(pipe.retriever.passages) if pipe else 0,
        "extractive_model": config.extractive_reader_path(),
        "generative_model": config.GEN_MODEL,
        "embed_model": pipe.retriever.model_name if pipe else None,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    pipe = state.get("pipe")
    if pipe is None:
        raise HTTPException(503, "Models are still loading")
    res = pipe.ask(req.question, reader=req.reader, top_k=req.top_k)
    data = res.to_dict()
    data.pop("source_passage")
    return data


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SQuAD QA Demo</title>
<style>
 body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:820px;margin:32px auto;padding:0 16px;color:#1d1d1f;background:#fafafa}
 h1{font-size:1.5rem;margin-bottom:4px} .sub{color:#666;margin-top:0}
 form{display:flex;gap:8px;flex-wrap:wrap;margin:20px 0}
 input[type=text]{flex:1 1 380px;padding:10px;font-size:1rem;border:1px solid #bbb;border-radius:6px}
 select,button{padding:10px;font-size:1rem;border-radius:6px;border:1px solid #bbb;background:#fff}
 button{background:#1f5fbf;color:#fff;border:none;cursor:pointer} button:disabled{opacity:.6}
 .answer{font-size:1.3rem;background:#fff;border-left:4px solid #1f5fbf;padding:12px 16px;border-radius:4px}
 .meta{color:#666;font-size:.9rem;margin:6px 0 18px}
 .psg{background:#fff;border:1px solid #e3e3e3;border-radius:6px;padding:10px 14px;margin:8px 0;font-size:.92rem;line-height:1.45}
 .psg b{color:#1f5fbf} mark{background:#ffe58a}
</style></head><body>
<h1>Question Answering in LLMs</h1>
<p class="sub">Sentence-Transformers + FAISS retriever over SQuAD passages &rarr; DistilBERT (extractive) or Flan-T5 (generative) reader</p>
<form id="f">
 <input type="text" id="q" value="Who was the Super Bowl 50 MVP?" required>
 <select id="r"><option value="extractive">DistilBERT (extractive)</option><option value="generative">Flan-T5 (generative)</option></select>
 <select id="k"><option>1</option><option selected>3</option><option>5</option></select>
 <button id="b">Ask</button>
</form>
<div id="out"></div>
<script>
const esc=s=>s.replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
document.getElementById("f").onsubmit=async e=>{
 e.preventDefault(); const b=document.getElementById("b"); b.disabled=true; b.textContent="Thinking...";
 const out=document.getElementById("out");
 try{
  const r=await fetch("/ask",{method:"POST",headers:{"Content-Type":"application/json"},
   body:JSON.stringify({question:document.getElementById("q").value,reader:document.getElementById("r").value,top_k:+document.getElementById("k").value})});
  const d=await r.json(); if(!r.ok) throw new Error(JSON.stringify(d.detail));
  const a=d.answer||"(no answer)";
  let h=`<div class="answer">${esc(a)}</div><div class="meta">${d.reader} reader &middot; source: ${esc(d.source_title||"-")} &middot; ${d.latency_ms.toFixed(0)} ms</div>`;
  d.retrieved.forEach((p,i)=>{let t=esc(p.text); if(d.answer){t=t.split(esc(d.answer)).join(`<mark>${esc(d.answer)}</mark>`);}
   h+=`<div class="psg"><b>[${i+1}] ${esc(p.title)}</b> &middot; similarity ${p.score.toFixed(3)}<br>${t}</div>`;});
  out.innerHTML=h;
 }catch(err){out.innerHTML=`<div class="psg">Error: ${esc(String(err))}</div>`;}
 b.disabled=false; b.textContent="Ask";
};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE
