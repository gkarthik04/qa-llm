"""SQuAD loading helpers and passage-corpus construction."""
from dataclasses import dataclass
from typing import List

from datasets import Dataset, load_dataset

from . import config


@dataclass
class Passage:
    pid: int
    title: str
    text: str


def load_squad(split: str) -> Dataset:
    return load_dataset(config.DATASET, split=split)


def build_corpus(ds: Dataset) -> List[Passage]:
    """Deduplicate the SQuAD contexts of a split into a passage list.

    SQuAD v1.1 validation has 10,570 questions over 2,067 unique paragraphs.
    """
    seen = {}
    passages: List[Passage] = []
    for title, context in zip(ds["title"], ds["context"]):
        if context not in seen:
            seen[context] = len(passages)
            passages.append(Passage(pid=len(passages), title=title.replace("_", " "), text=context))
    return passages


def gold_passage_ids(ds: Dataset, passages: List[Passage]) -> List[int]:
    lookup = {p.text: p.pid for p in passages}
    return [lookup[c] for c in ds["context"]]


def sample(ds: Dataset, n: int, seed: int = config.SEED) -> Dataset:
    """Deterministic random subset (or the full split if n <= 0 or n >= len)."""
    if n <= 0 or n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))
