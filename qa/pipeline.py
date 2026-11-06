"""Retrieve-then-read QA pipeline."""
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List

from .readers import load_reader
from .retriever import DenseRetriever


@dataclass
class QAResult:
    question: str
    answer: str
    reader: str
    score: float | None
    source_title: str | None
    source_passage: str | None
    retrieved: List[Dict] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class QAPipeline:
    """Holds one retriever and lazily-loaded readers so the CLI/API can switch readers."""

    def __init__(self, retriever: DenseRetriever | None = None):
        self.retriever = retriever or DenseRetriever.load()
        self._readers = {}

    def reader(self, kind: str):
        if kind not in self._readers:
            self._readers[kind] = load_reader(kind)
        return self._readers[kind]

    def ask(self, question: str, reader: str = "extractive", top_k: int = 3) -> QAResult:
        t0 = time.perf_counter()
        hits = self.retriever.search([question], k=top_k)[0]
        contexts = [p.text for p, _ in hits]
        ans = self.reader(reader).read(question, contexts)

        # The extractive reader tells us which passage the span came from; for the
        # generative reader we credit the passage that contains the answer, else the top hit.
        idx = ans.passage_index
        if idx is None:
            idx = next((i for i, c in enumerate(contexts) if ans.text and ans.text.lower() in c.lower()), 0)
        src = hits[idx][0] if hits else None
        return QAResult(
            question=question,
            answer=ans.text,
            reader=reader,
            score=ans.score,
            source_title=src.title if src else None,
            source_passage=src.text if src else None,
            retrieved=[{"pid": p.pid, "title": p.title, "score": round(s, 4), "text": p.text} for p, s in hits],
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
