"""Okapi BM25 sparse retriever (pure NumPy, no extra dependency).

Lexical matching complements dense embeddings: rare names, numbers and exact
phrases ("Super Bowl 50", "1852") are where MiniLM embeddings are weakest.
"""
import re
from collections import Counter
from typing import List, Sequence, Tuple

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by did do does for from had has have how in is it its of on or that the "
    "their there this to was were what when where which who whom whose why will with".split()
)


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vocab = {}
        rows, cols, tfs = [], [], []
        lengths = []
        for d, doc in enumerate(documents):
            counts = Counter(tokenize(doc))
            lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                idx = self.vocab.setdefault(term, len(self.vocab))
                rows.append(idx)
                cols.append(d)
                tfs.append(tf)
        self.n_docs = len(documents)
        self.doc_len = np.array(lengths, dtype=np.float32)
        avg = self.doc_len.mean() if self.n_docs else 1.0
        # Postings as one array per term: (doc ids, precomputed term weights).
        rows = np.array(rows, dtype=np.int64)
        cols = np.array(cols, dtype=np.int64)
        tfs = np.array(tfs, dtype=np.float32)
        norm = tfs * (k1 + 1) / (tfs + k1 * (1 - b + b * self.doc_len[cols] / avg))
        order = np.argsort(rows, kind="stable")
        rows, cols, norm = rows[order], cols[order], norm[order]
        bounds = np.searchsorted(rows, np.arange(len(self.vocab) + 1))
        self.postings = [(cols[bounds[i]:bounds[i + 1]], norm[bounds[i]:bounds[i + 1]]) for i in range(len(self.vocab))]
        df = np.diff(bounds).astype(np.float32)
        self.idf = np.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float32)
        for term in set(tokenize(query)):
            idx = self.vocab.get(term)
            if idx is not None:
                docs, w = self.postings[idx]
                out[docs] += self.idf[idx] * w
        return out

    def search(self, query: str, k: int) -> List[Tuple[int, float]]:
        s = self.scores(query)
        k = min(k, self.n_docs)
        top = np.argpartition(-s, k - 1)[:k] if k < self.n_docs else np.arange(self.n_docs)
        top = top[np.argsort(-s[top])]
        return [(int(i), float(s[i])) for i in top if s[i] > 0]
