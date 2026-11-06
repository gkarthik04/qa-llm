"""Bring-your-own-documents: extract text from PDF / TXT / MD uploads, chunk it into
overlapping passages and index them on the fly (dense + BM25), so the same pipeline
answers questions about the user's own files.
"""
import io
import re
import threading
import uuid
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from . import config
from .data import Passage
from .search import SearchEngine, embed_passages

SUPPORTED = (".pdf", ".txt", ".md")


class UploadError(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception as exc:  # malformed / encrypted PDFs
            raise UploadError(f"Could not read PDF: {exc}") from exc
        text = "\n".join(pages)
    elif name.endswith((".txt", ".md")):
        text = data.decode("utf-8", errors="replace")
    else:
        raise UploadError(f"Unsupported file type. Use one of: {', '.join(SUPPORTED)}")
    text = re.sub(r"-\n(\w)", r"\1", text)        # re-join words hyphenated across lines
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text).strip()
    if len(text) < 20:
        raise UploadError("No extractable text found (scanned PDFs need OCR first).")
    return text


def chunk_text(text: str, words: int = config.CHUNK_WORDS, overlap: int = config.CHUNK_OVERLAP) -> List[str]:
    """Sentence-aware sliding window of roughly `words` words with `overlap` words carried over."""
    sentences = re.split(r"(?<=[.!?])\s+|\n\n", text)
    chunks, current, count = [], [], 0
    for sent in sentences:
        sent = " ".join(sent.split())
        if not sent:
            continue
        n = len(sent.split())
        if current and count + n > words:
            chunks.append(" ".join(current))
            carry, c = [], 0
            for s in reversed(current):
                c += len(s.split())
                carry.insert(0, s)
                if c >= overlap:
                    break
            current, count = carry, sum(len(s.split()) for s in carry)
        current.append(sent)
        count += n
    if current:
        chunks.append(" ".join(current))
    # very long "sentences" (tables, missing punctuation) are split hard
    out = []
    for ch in chunks:
        toks = ch.split()
        for i in range(0, len(toks), words * 2):
            out.append(" ".join(toks[i:i + words * 2]))
    return out


@dataclass
class Document:
    doc_id: str
    name: str
    chars: int
    chunks: int


class DocumentStore:
    """In-memory store of uploaded documents with its own SearchEngine (rebuilt on change)."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.docs: Dict[str, Document] = {}
        self._chunks: Dict[str, List[str]] = {}
        self._emb: Dict[str, np.ndarray] = {}
        self.engine: SearchEngine | None = None
        self._lock = threading.Lock()

    def add(self, filename: str, data: bytes) -> Document:
        if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
            raise UploadError(f"File larger than {config.MAX_UPLOAD_MB} MB")
        text = extract_text(filename, data)
        chunks = chunk_text(text)
        with self._lock:
            total = sum(len(c) for c in self._chunks.values()) + len(chunks)
            if total > config.MAX_UPLOAD_CHUNKS:
                raise UploadError(f"Upload limit reached ({config.MAX_UPLOAD_CHUNKS} passages); remove a document first")
            doc = Document(uuid.uuid4().hex[:8], filename, len(text), len(chunks))
            passages = [Passage(0, f"{filename} · part {j + 1}", ch) for j, ch in enumerate(chunks)]
            self._emb[doc.doc_id] = embed_passages(passages, self.encoder)  # embed once per document
            self.docs[doc.doc_id] = doc
            self._chunks[doc.doc_id] = chunks
            self._rebuild()
        return doc

    def remove(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self.docs:
                return False
            for d in (self.docs, self._chunks, self._emb):
                del d[doc_id]
            self._rebuild()
        return True

    def clear(self):
        with self._lock:
            for d in (self.docs, self._chunks, self._emb):
                d.clear()
            self.engine = None

    def _rebuild(self):
        passages = []
        for doc_id, chunks in self._chunks.items():
            name = self.docs[doc_id].name
            for j, ch in enumerate(chunks):
                passages.append(Passage(pid=len(passages), title=f"{name} · part {j + 1}", text=ch))
        if not passages:
            self.engine = None
            return
        emb = np.concatenate([self._emb[d] for d in self._chunks])
        self.engine = SearchEngine.from_passages(passages, self.encoder, embeddings=emb)

    def list(self) -> List[dict]:
        return [d.__dict__ for d in self.docs.values()]
