"""Answer readers.

* ``ExtractiveReader`` - DistilBERT span extraction (start/end logits) over one or
  more passages; long passages are split into overlapping windows.
* ``GenerativeReader`` - Flan-T5 that writes the answer given the passages in a prompt.

Both expose ``read_batch(questions, contexts_per_question) -> List[Answer]``.
"""
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, AutoTokenizer

from . import config
from .metrics import normalize_answer


@dataclass
class Answer:
    text: str
    score: float | None = None
    passage_index: int | None = None  # which of the supplied contexts the answer came from
    confidence: float | None = None   # probability mass of this answer among the candidates (0-1)
    char_start: int | None = None     # span offsets inside contexts[passage_index] (extractive only)
    alternatives: List[dict] = field(default_factory=list)  # runner-up answers with their confidence


def _log_softmax(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = np.where(mask, x, -np.inf)
    m = x[mask].max()
    return x - (m + np.log(np.exp(x[mask] - m).sum()))


def _aggregate(candidates, top_n: int = 5):
    """Turn (score, text, passage, char_start) candidates into probabilities.

    Softmax over all candidate spans, then sum the mass of spans with the same
    normalised text (the same answer is often found in several windows/passages).
    """
    if not candidates:
        return None, []
    scores = np.array([c[0] for c in candidates])
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    mass, best_of = {}, {}
    for p, c in zip(probs, candidates):
        key = normalize_answer(c[1]) or c[1]
        mass[key] = mass.get(key, 0.0) + float(p)
        if key not in best_of or c[0] > best_of[key][0]:
            best_of[key] = c
    ranked = sorted(mass, key=lambda k: -best_of[k][0])
    alts = [{"text": best_of[k][1], "confidence": round(mass[k], 4), "passage_index": best_of[k][2]}
            for k in ranked[:top_n]]
    return mass[ranked[0]], alts


class ExtractiveReader:
    """DistilBERT span reader.

    scoring="raw"        - span score = start_logit + end_logit (the usual HF recipe). Raw logits are not
                           comparable across windows/passages, which lets a distractor passage win.
    scoring="calibrated" - span score = log P(start) + log P(end) within its window (softmax over the
                           window's context tokens) + log P(passage | question) from the retriever /
                           re-ranker. Every window is turned into a proper distribution first, and the
                           retrieval evidence is folded in.
    """
    name = "extractive"

    def __init__(self, model_path: str | None = None, max_length: int = config.MAX_LENGTH,
                 stride: int = config.DOC_STRIDE, max_answer_tokens: int = config.MAX_ANSWER_TOKENS,
                 batch_size: int = 32, n_best: int = 20, scoring: str = "raw"):
        self.model_path = model_path or config.extractive_reader_path()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_path).eval()
        self.max_length = max_length
        self.stride = stride
        self.max_answer_tokens = max_answer_tokens
        self.batch_size = batch_size
        self.n_best = n_best
        if scoring not in ("raw", "calibrated"):
            raise ValueError("scoring must be 'raw' or 'calibrated'")
        self.scoring = scoring

    @torch.inference_mode()
    def read_batch(self, questions: Sequence[str], contexts: Sequence[Sequence[str]],
                   priors: Sequence[Sequence[float]] | None = None, scoring: str | None = None) -> List[Answer]:
        scoring = scoring or self.scoring
        pair_q, pair_c, pair_owner = [], [], []
        for qi, (q, ctxs) in enumerate(zip(questions, contexts)):
            for ci, c in enumerate(ctxs):
                pair_q.append(q.strip())
                pair_c.append(c)
                pair_owner.append((qi, ci))

        enc = self.tokenizer(
            pair_q, pair_c,
            truncation="only_second", max_length=self.max_length, stride=self.stride,
            return_overflowing_tokens=True, return_offsets_mapping=True,
        )
        feature_to_pair = enc["overflow_to_sample_mapping"]
        offsets = enc["offset_mapping"]
        n_feat = len(feature_to_pair)

        # Pad per mini-batch (dynamic padding) rather than to max_length: much faster on CPU.
        start_logits, end_logits = [], []
        for i in range(0, n_feat, self.batch_size):
            batch = self.tokenizer.pad(
                {"input_ids": enc["input_ids"][i:i + self.batch_size],
                 "attention_mask": enc["attention_mask"][i:i + self.batch_size]},
                return_tensors="pt",
            )
            out = self.model(**batch)
            lengths = batch["attention_mask"].sum(dim=1).tolist()
            for row, n in enumerate(lengths):
                start_logits.append(out.start_logits[row, :n].numpy())
                end_logits.append(out.end_logits[row, :n].numpy())

        candidates = [[] for _ in questions]  # (score, text, passage_index, char_start)
        for f in range(n_feat):
            qi, ci = pair_owner[feature_to_pair[f]]
            seq_ids = enc.sequence_ids(f)
            ctx_mask = np.array([s == 1 for s in seq_ids])
            if not ctx_mask.any():
                continue
            if scoring == "calibrated":
                s_log = _log_softmax(start_logits[f], ctx_mask)
                e_log = _log_softmax(end_logits[f], ctx_mask)
                bonus = float(np.log(max(priors[qi][ci], 1e-9))) if priors is not None else 0.0
            else:
                s_log = np.where(ctx_mask, start_logits[f], -np.inf)
                e_log = np.where(ctx_mask, end_logits[f], -np.inf)
                bonus = 0.0
            top_s = np.argsort(s_log)[::-1][: self.n_best]
            top_e = np.argsort(e_log)[::-1][: self.n_best]
            context = pair_c[feature_to_pair[f]]
            for s in top_s:
                if not ctx_mask[s]:
                    continue
                for e in top_e:
                    if not ctx_mask[e] or e < s or e - s + 1 > self.max_answer_tokens:
                        continue
                    char_s, char_e = offsets[f][s][0], offsets[f][e][1]
                    raw = context[char_s:char_e]
                    text = raw.strip()
                    lead = len(raw) - len(raw.lstrip())
                    candidates[qi].append((float(s_log[s] + e_log[e]) + bonus, text, ci, char_s + lead))

        answers = []
        for cands in candidates:
            if not cands:
                answers.append(Answer(text=""))
                continue
            best = max(cands, key=lambda c: c[0])
            conf, alts = _aggregate(cands)
            answers.append(Answer(text=best[1], score=best[0], passage_index=best[2], char_start=best[3],
                                  confidence=conf, alternatives=alts))
        return answers

    def read(self, question: str, contexts: Sequence[str], priors: Sequence[float] | None = None,
             scoring: str | None = None) -> Answer:
        return self.read_batch([question], [contexts], [priors] if priors is not None else None, scoring)[0]


class GenerativeReader:
    name = "generative"
    PROMPT = (
        "Read the passages and answer the question with a short phrase copied from them.\n\n"
        "{passages}\n\nQuestion: {question}\nAnswer:"
    )

    def __init__(self, model_name: str = config.GEN_MODEL, max_input_tokens: int = 512,
                 max_new_tokens: int = 32, batch_size: int = 8):
        self.model_path = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    def _fit(self, text: str, max_tokens: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        return text if len(ids) <= max_tokens else self.tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)

    def build_prompt(self, question: str, contexts: Sequence[str]) -> str:
        """Fill the template with as many passages (in rank order) as fit in the token budget.

        Plain right-truncation of the whole prompt would cut off the question (it sits at
        the end), so the passages are trimmed instead and the question is always kept.
        Higher-ranked passages are kept whole; the last one that fits is trimmed.
        """
        question = question.strip()
        labels = ["Passage:"] if len(contexts) == 1 else [f"Passage {i + 1}:" for i in range(len(contexts))]
        skeleton = self.PROMPT.format(passages="\n\n".join(labels), question=question)
        budget = self.max_input_tokens - len(self.tokenizer(skeleton).input_ids) - 4
        parts = []
        for lab, c in zip(labels, contexts):
            if budget < 16:
                break
            text = self._fit(c, budget)
            budget -= len(self.tokenizer(text, add_special_tokens=False).input_ids)
            parts.append(f"{lab} {text}")
        return self.PROMPT.format(passages="\n\n".join(parts), question=question)

    @torch.inference_mode()
    def read_batch(self, questions: Sequence[str], contexts: Sequence[Sequence[str]], priors=None,
                   scoring=None) -> List[Answer]:
        """`priors` / `scoring` are accepted for interface parity with ExtractiveReader and ignored.

        Confidence = probability of the generated answer under greedy decoding
        (product of the chosen tokens' probabilities).
        """
        prompts = [self.build_prompt(q, c) for q, c in zip(questions, contexts)]
        answers: List[Answer] = []
        for i in range(0, len(prompts), self.batch_size):
            enc = self.tokenizer(prompts[i:i + self.batch_size], return_tensors="pt", padding=True,
                                 truncation=True, max_length=self.max_input_tokens)
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, num_beams=1, do_sample=False,
                                      return_dict_in_generate=True, output_scores=True)
            step_lp = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
            gen = out.sequences[:, 1:]  # drop the decoder start token
            pad = self.tokenizer.pad_token_id
            for row, text in enumerate(self.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)):
                keep = gen[row] != pad
                logp = float(step_lp[row][keep].sum())
                answers.append(Answer(text=text.strip(), score=logp, confidence=float(np.exp(logp))))
        for a, ctxs in zip(answers, contexts):
            # Credit the first passage that contains the generated answer (None if it was not copied).
            low = a.text.lower()
            for ci, c in enumerate(ctxs):
                pos = c.lower().find(low) if low else -1
                if pos >= 0:
                    a.passage_index, a.char_start = ci, pos
                    break
        return answers

    def read(self, question: str, contexts: Sequence[str], priors=None, scoring=None) -> Answer:
        return self.read_batch([question], [contexts])[0]


def load_reader(kind: str, model_path: str | None = None, scoring: str = "raw"):
    if kind == "extractive":
        return ExtractiveReader(model_path, scoring=scoring)
    if kind == "generative":
        return GenerativeReader(model_path or config.GEN_MODEL)
    raise ValueError(f"Unknown reader '{kind}' (expected 'extractive' or 'generative')")
