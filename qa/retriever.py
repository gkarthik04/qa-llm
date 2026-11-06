"""Dense passage retriever: Sentence-Transformers embeddings + FAISS inner-product index."""
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from . import config
from .data import Passage


class DenseRetriever:
    def __init__(self, model_name: str = config.EMBED_MODEL):
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name, device="cpu")
        self.index: faiss.Index | None = None
        self.passages: List[Passage] = []

    def _embed(self, texts: Sequence[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        emb = self.encoder.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine similarity == inner product
            show_progress_bar=show_progress,
        )
        return emb.astype("float32")

    def build(self, passages: List[Passage], show_progress: bool = True) -> None:
        self.passages = passages
        # Prefix the article title: it often carries the entity the question asks about.
        texts = [f"{p.title}. {p.text}" for p in passages]
        emb = self._embed(texts, show_progress=show_progress)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def save(self, directory: Path = config.INDEX_DIR) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "passages.faiss"))
        with open(directory / "passages.jsonl", "w", encoding="utf-8") as f:
            for p in self.passages:
                f.write(json.dumps(p.__dict__, ensure_ascii=False) + "\n")
        meta = {"embed_model": self.model_name, "num_passages": len(self.passages), "dim": self.index.d}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path = config.INDEX_DIR) -> "DenseRetriever":
        directory = Path(directory)
        if not (directory / "passages.faiss").exists():
            raise FileNotFoundError(f"No index in {directory}. Run `python -m qa.build_index` first.")
        meta = json.loads((directory / "meta.json").read_text())
        retriever = cls(meta["embed_model"])
        retriever.index = faiss.read_index(str(directory / "passages.faiss"))
        with open(directory / "passages.jsonl", encoding="utf-8") as f:
            retriever.passages = [Passage(**json.loads(line)) for line in f]
        return retriever

    def search(self, queries: Sequence[str], k: int = 5, batch_size: int = 64) -> List[List[Tuple[Passage, float]]]:
        emb = self._embed(queries, batch_size=batch_size)
        scores, ids = self.index.search(emb, k)
        return [
            [(self.passages[i], float(s)) for i, s in zip(row_ids, row_scores) if i != -1]
            for row_ids, row_scores in zip(ids, scores)
        ]
