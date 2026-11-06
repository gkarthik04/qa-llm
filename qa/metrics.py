"""SQuAD v1.1 metrics (Exact Match, token F1) and retrieval Recall@k.

The normalisation follows the official SQuAD v1.1 evaluation script:
lower-case, strip punctuation, drop articles (a/an/the), collapse whitespace.
"""
import re
import string
from collections import Counter
from typing import Iterable, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(truth))


def f1_score(prediction: str, truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def max_over_truths(metric, prediction: str, truths: Iterable[str]) -> float:
    """SQuAD has several reference answers per question; take the best match."""
    return max(metric(prediction, t) for t in truths)


def squad_scores(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> dict:
    """Corpus-level EM / F1 in percent."""
    assert len(predictions) == len(references)
    n = len(predictions)
    em = sum(max_over_truths(exact_match, p, r) for p, r in zip(predictions, references))
    f1 = sum(max_over_truths(f1_score, p, r) for p, r in zip(predictions, references))
    return {"exact_match": 100.0 * em / n, "f1": 100.0 * f1 / n, "n": n}


def recall_at_k(retrieved_ids: Sequence[Sequence[int]], gold_ids: Sequence[int], ks: Sequence[int]) -> dict:
    """Fraction (percent) of questions whose gold passage is in the top-k results."""
    out = {}
    n = len(gold_ids)
    for k in ks:
        hits = sum(1 for ret, gold in zip(retrieved_ids, gold_ids) if gold in list(ret)[:k])
        out[f"recall@{k}"] = 100.0 * hits / n
    return out


def mean_reciprocal_rank(retrieved_ids: Sequence[Sequence[int]], gold_ids: Sequence[int]) -> float:
    total = 0.0
    for ret, gold in zip(retrieved_ids, gold_ids):
        ret = list(ret)
        if gold in ret:
            total += 1.0 / (ret.index(gold) + 1)
    return total / len(gold_ids)
