"""Search engine on top of a passage collection: dense, BM25, hybrid (RRF) and cross-encoder re-ranking.

    dense   - MiniLM bi-encoder + FAISS (cosine)
    bm25    - Okapi BM25 lexical scores
    hybrid  - Reciprocal Rank Fusion of the dense and BM25 rankings
    rerank  - optional: a cross-encoder rescores the top `candidates` of any of the above

Each hit also carries a `prior`, a probability for "this passage answers the question"
(softmax over the returned hits). The calibrated extractive reader uses it.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from . import config
from .bm25 import BM25
from .data import Passage

MODES = ("dense", "bm25", "hybrid")
RRF_K = 60            # standard constant from Cormack et al. (2009)
DENSE_TEMPERATURE = 0.05


@dataclass
class Hit:
    passage: Passage
    score: float                 # score used for the final ordering
    dense: float | None = None   # cosine similarity
    bm25: float | None = None
    rerank: float | None = None  # cross-encoder logit
    prior: float = 0.0           # P(passage is relevant), softmax over returned hits

    def to_dict(self) -> dict:
        p = self.passage
        return {"pid": p.pid, "title": p.title, "text": p.text, "score": round(self.score, 4),
                "dense": _r(self.dense), "bm25": _r(self.bm25), "rerank": _r(self.rerank),
                "prior": round(self.prior, 4)}


def _r(x):
    return None if x is None else round(float(x), 4)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def embed_passages(passages: Sequence[Passage], encoder) -> np.ndarray:
    return encoder.encode([f"{p.title}. {p.text}" for p in passages], batch_size=32,
                          convert_to_numpy=True, normalize_embeddings=True).astype("float32")


_reranker = None


def get_reranker():
    """Cross-encoder shared by every engine (loaded on first use)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(config.RERANK_MODEL, device="cpu")
    return _reranker


@dataclass
class SearchEngine:
    passages: List[Passage]
    index: object            # faiss index over normalised embeddings
    encoder: object          # SentenceTransformer
    bm25: BM25 = field(init=False)

    def __post_init__(self):
        self.bm25 = BM25([f"{p.title} {p.text}" for p in self.passages])

    @classmethod
    def from_retriever(cls, retriever) -> "SearchEngine":
        return cls(retriever.passages, retriever.index, retriever.encoder)

    @classmethod
    def from_passages(cls, passages: List[Passage], encoder, embeddings: np.ndarray | None = None) -> "SearchEngine":
        import faiss
        emb = embeddings if embeddings is not None else embed_passages(passages, encoder)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        return cls(passages, index, encoder)

    def __len__(self):
        return len(self.passages)

    def _dense_vectors(self, ids: Sequence[int]) -> np.ndarray:
        return np.stack([self.index.reconstruct(int(i)) for i in ids])

    def search_batch(self, queries: Sequence[str], k: int = 3, mode: str = "hybrid", rerank: bool = False,
                     candidates: int = 20, timings: Dict[str, float] | None = None) -> List[List[Hit]]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        n = len(self.passages)
        pool = min(n, max(k, candidates if (rerank or mode == "hybrid") else k))
        t0 = time.perf_counter()
        q_emb = self.encoder.encode(list(queries), batch_size=64, convert_to_numpy=True,
                                    normalize_embeddings=True).astype("float32")
        d_scores, d_ids = self.index.search(q_emb, pool)

        results: List[List[Hit]] = []
        for qi, q in enumerate(queries):
            dense = {int(i): float(s) for i, s in zip(d_ids[qi], d_scores[qi]) if i != -1}
            if mode == "dense":
                ranked = sorted(dense.items(), key=lambda x: -x[1])
                hits = [Hit(self.passages[i], s, dense=s) for i, s in ranked]
            else:
                sparse = dict(self.bm25.search(q, pool))
                if mode == "bm25":
                    order = sorted(sparse.items(), key=lambda x: -x[1])
                    fused = {i: s for i, s in order}
                else:
                    fused = {}
                    for ranking in (sorted(dense, key=lambda i: -dense[i]), sorted(sparse, key=lambda i: -sparse[i])):
                        for rank, i in enumerate(ranking):
                            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
                # fill in dense similarity for passages only BM25 found (needed for the prior)
                missing = [i for i in fused if i not in dense]
                if missing:
                    sims = self._dense_vectors(missing) @ q_emb[qi]
                    dense.update({i: float(s) for i, s in zip(missing, sims)})
                ranked = sorted(fused.items(), key=lambda x: -x[1])[:pool]
                hits = [Hit(self.passages[i], s, dense=dense.get(i), bm25=sparse.get(i, 0.0)) for i, s in ranked]
            results.append(hits)
        if timings is not None:
            timings["retrieve_ms"] = 1000 * (time.perf_counter() - t0)

        if rerank:
            t1 = time.perf_counter()
            pairs = [(q, h.passage.text) for q, hits in zip(queries, results) for h in hits]
            if pairs:
                logits = get_reranker().predict(pairs, batch_size=64, show_progress_bar=False)
                j = 0
                for hits in results:
                    for h in hits:
                        h.rerank = float(logits[j])
                        h.score = h.rerank
                        j += 1
                    hits.sort(key=lambda h: -h.score)
            if timings is not None:
                timings["rerank_ms"] = 1000 * (time.perf_counter() - t1)

        for hits in results:
            del hits[k:]
            if not hits:
                continue
            if rerank:
                prior = _softmax(np.array([h.rerank for h in hits]))
            else:
                prior = _softmax(np.array([h.dense for h in hits]) / DENSE_TEMPERATURE)
            for h, p in zip(hits, prior):
                h.prior = float(p)
        return results

    def search(self, query: str, **kwargs) -> List[Hit]:
        return self.search_batch([query], **kwargs)[0]
