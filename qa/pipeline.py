"""Retrieve-then-read QA pipeline.

    retrieval: dense | bm25 | hybrid   (+ optional cross-encoder re-ranking)
    reader:    extractive (DistilBERT, raw or calibrated scoring) | generative (Flan-T5)
    corpus:    squad (pre-built index) | docs (files uploaded at runtime)

`compare()` runs both readers on the same evidence and reports whether they agree;
agreement is a strong, model-free confidence signal (see results/advanced.json).
"""
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List

from . import config
from .documents import DocumentStore
from .metrics import f1_score
from .readers import load_reader
from .retriever import DenseRetriever
from .search import SearchEngine

AGREE_F1 = 0.5  # token-F1 between the two readers' answers above which they "agree"


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
    confidence: float | None = None
    passage_index: int | None = None
    char_start: int | None = None
    alternatives: List[Dict] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    settings: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class QAPipeline:
    """Holds the SQuAD search engine, an uploaded-documents store and lazily-loaded readers."""

    def __init__(self, retriever: DenseRetriever | None = None):
        self.retriever = retriever or DenseRetriever.load()
        self.engine = SearchEngine.from_retriever(self.retriever)
        self.documents = DocumentStore(self.retriever.encoder)
        self._readers = {}

    def reader(self, kind: str):
        if kind not in self._readers:
            self._readers[kind] = load_reader(kind)
        return self._readers[kind]

    def engine_for(self, corpus: str) -> SearchEngine:
        if corpus == "squad":
            return self.engine
        if corpus == "docs":
            if self.documents.engine is None:
                raise LookupError("No documents uploaded yet")
            return self.documents.engine
        raise ValueError("corpus must be 'squad' or 'docs'")

    def retrieve(self, question: str, top_k: int = 3, retrieval: str = config.DEFAULT_RETRIEVAL,
                 rerank: bool = config.DEFAULT_RERANK, corpus: str = "squad", timings: Dict | None = None):
        return self.engine_for(corpus).search(question, k=top_k, mode=retrieval, rerank=rerank, timings=timings)

    def _read(self, question, hits, reader, scoring, timings, settings, t0) -> QAResult:
        t1 = time.perf_counter()
        contexts = [h.passage.text for h in hits]
        ans = self.reader(reader).read(question, contexts, priors=[h.prior for h in hits], scoring=scoring)
        timings["read_ms"] = 1000 * (time.perf_counter() - t1)
        idx = ans.passage_index
        src = hits[idx].passage if (idx is not None and hits) else (hits[0].passage if hits else None)
        return QAResult(
            question=question, answer=ans.text, reader=reader, score=ans.score,
            source_title=src.title if src else None, source_passage=src.text if src else None,
            retrieved=[h.to_dict() for h in hits],
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            confidence=ans.confidence, passage_index=idx, char_start=ans.char_start,
            alternatives=ans.alternatives, timings={k: round(v, 1) for k, v in timings.items()},
            settings=settings,
        )

    def ask(self, question: str, reader: str = "extractive", top_k: int = 3,
            retrieval: str = config.DEFAULT_RETRIEVAL, rerank: bool = config.DEFAULT_RERANK,
            scoring: str = config.DEFAULT_SCORING, corpus: str = "squad") -> QAResult:
        t0 = time.perf_counter()
        timings: Dict[str, float] = {}
        hits = self.retrieve(question, top_k, retrieval, rerank, corpus, timings)
        settings = dict(top_k=top_k, retrieval=retrieval, rerank=rerank, scoring=scoring, corpus=corpus)
        return self._read(question, hits, reader, scoring, timings, settings, t0)

    def compare(self, question: str, top_k: int = 3, retrieval: str = config.DEFAULT_RETRIEVAL,
                rerank: bool = config.DEFAULT_RERANK, scoring: str = config.DEFAULT_SCORING,
                corpus: str = "squad") -> dict:
        """Both readers on the same retrieved evidence, plus an agreement verdict."""
        t0 = time.perf_counter()
        timings: Dict[str, float] = {}
        hits = self.retrieve(question, top_k, retrieval, rerank, corpus, timings)
        settings = dict(top_k=top_k, retrieval=retrieval, rerank=rerank, scoring=scoring, corpus=corpus)
        ext = self._read(question, hits, "extractive", scoring, dict(timings), settings, t0)
        gen = self._read(question, hits, "generative", scoring, dict(timings), settings, t0)
        overlap = f1_score(ext.answer, gen.answer) if ext.answer and gen.answer else 0.0
        return {
            "question": question,
            "extractive": ext.to_dict(),
            "generative": gen.to_dict(),
            "agreement": {"f1": round(overlap, 3), "agree": overlap >= AGREE_F1},
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
