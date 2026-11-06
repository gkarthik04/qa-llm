Measured on `12th Gen Intel(R) Core(TM) i5-1235U, 10 threads` (CPU only, 2026-11-06).

**Retriever** - `sentence-transformers/all-MiniLM-L6-v2` + FAISS `IndexFlatIP` over 2,067 SQuAD v1.1 validation paragraphs, 10,570 questions:

| Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@20 | MRR@20 | ms / query |
|---:|---:|---:|---:|---:|---:|---:|
| 62.3 | 80.5 | 86.5 | 91.7 | 95.3 | 72.8 | 2.0 |

**Readers** - SQuAD v1.1 validation, EM / F1 in %:

| Reader | Type | Setting | Questions | EM | F1 | ms / question |
|---|---|---|---:|---:|---:|---:|
| DistilBERT (pretrained, not fine-tuned by us) | extractive | oracle passage | 1,000 | 79.10 | 86.79 | 173 |
| DistilBERT (pretrained, not fine-tuned by us) | extractive | retrieved top-3 | 500 | 57.00 | 65.14 | 558 |
| DistilBERT (fine-tuned by us) | extractive | oracle passage | 1,000 | 77.70 | 86.34 | 190 |
| DistilBERT (fine-tuned by us) | extractive | retrieved top-3 | 500 | 56.60 | 65.58 | 573 |
| Flan-T5 (flan-t5-base, zero-shot) | generative | oracle passage | 1,000 | 84.40 | 91.34 | 723 |
| Flan-T5 (flan-t5-base, zero-shot) | generative | retrieved top-3 | 500 | 62.00 | 68.65 | 1328 |
