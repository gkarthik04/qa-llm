"""Answer readers.

* ``ExtractiveReader`` - DistilBERT span extraction (start/end logits) over one or
  more passages; long passages are split into overlapping windows.
* ``GenerativeReader`` - Flan-T5 that writes the answer given the passages in a prompt.

Both expose ``read_batch(questions, contexts_per_question) -> List[Answer]``.
"""
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, AutoTokenizer

from . import config


@dataclass
class Answer:
    text: str
    score: float | None = None
    passage_index: int | None = None  # which of the supplied contexts the answer came from


class ExtractiveReader:
    name = "extractive"

    def __init__(self, model_path: str | None = None, max_length: int = config.MAX_LENGTH,
                 stride: int = config.DOC_STRIDE, max_answer_tokens: int = config.MAX_ANSWER_TOKENS,
                 batch_size: int = 32, n_best: int = 20):
        self.model_path = model_path or config.extractive_reader_path()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_path).eval()
        self.max_length = max_length
        self.stride = stride
        self.max_answer_tokens = max_answer_tokens
        self.batch_size = batch_size
        self.n_best = n_best

    @torch.inference_mode()
    def read_batch(self, questions: Sequence[str], contexts: Sequence[Sequence[str]]) -> List[Answer]:
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

        best = [Answer(text="", score=float("-inf")) for _ in questions]
        for f in range(n_feat):
            qi, ci = pair_owner[feature_to_pair[f]]
            seq_ids = enc.sequence_ids(f)
            ctx_mask = np.array([s == 1 for s in seq_ids])
            s_log = np.where(ctx_mask, start_logits[f], -1e9)
            e_log = np.where(ctx_mask, end_logits[f], -1e9)
            top_s = np.argsort(s_log)[::-1][: self.n_best]
            top_e = np.argsort(e_log)[::-1][: self.n_best]
            for s in top_s:
                if not ctx_mask[s]:
                    continue
                for e in top_e:
                    if not ctx_mask[e] or e < s or e - s + 1 > self.max_answer_tokens:
                        continue
                    score = float(s_log[s] + e_log[e])
                    if score > best[qi].score:
                        char_s, char_e = offsets[f][s][0], offsets[f][e][1]
                        best[qi] = Answer(text=pair_c[feature_to_pair[f]][char_s:char_e].strip(),
                                          score=score, passage_index=ci)
        for a in best:
            if a.score == float("-inf"):
                a.score = None
        return best

    def read(self, question: str, contexts: Sequence[str]) -> Answer:
        return self.read_batch([question], [contexts])[0]


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
    def read_batch(self, questions: Sequence[str], contexts: Sequence[Sequence[str]]) -> List[Answer]:
        prompts = [self.build_prompt(q, c) for q, c in zip(questions, contexts)]
        answers: List[Answer] = []
        for i in range(0, len(prompts), self.batch_size):
            enc = self.tokenizer(prompts[i:i + self.batch_size], return_tensors="pt", padding=True,
                                 truncation=True, max_length=self.max_input_tokens)
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, num_beams=1, do_sample=False)
            answers.extend(Answer(text=t.strip()) for t in self.tokenizer.batch_decode(out, skip_special_tokens=True))
        return answers

    def read(self, question: str, contexts: Sequence[str]) -> Answer:
        return self.read_batch([question], [contexts])[0]


def load_reader(kind: str, model_path: str | None = None):
    if kind == "extractive":
        return ExtractiveReader(model_path)
    if kind == "generative":
        return GenerativeReader(model_path or config.GEN_MODEL)
    raise ValueError(f"Unknown reader '{kind}' (expected 'extractive' or 'generative')")
