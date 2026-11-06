Measured on `12th Gen Intel(R) Core(TM) i5-1235U, 10 threads` (CPU only, 2026-11-06). Retrieval: 1,000 questions; readers: 500 questions (the same seeded sample as the baseline table), top-3 passages.

**A. Retrieval variants**

| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | ms / query |
|---|---:|---:|---:|---:|---:|---:|
| Dense (MiniLM + FAISS) | 60.7 | 79.5 | 86.0 | 90.9 | 71.2 | 3 |
| BM25 | 77.5 | 89.2 | 91.9 | 94.4 | 83.8 | 3 |
| Hybrid (RRF dense + BM25) | 70.4 | 87.6 | 93.1 | 96.7 | 80.1 | 3 |
| Hybrid + cross-encoder re-rank (top 20) | 87.6 | 96.1 | 97.5 | 98.3 | 91.9 | 653 |

**B. Full pipeline (retrieve top-3 -> read)**

| Retrieval | Reader | Scoring | EM | F1 | ms / question (read) |
|---|---|---|---:|---:|---:|
| dense | DistilBERT (fine-tuned by us) | raw | 56.60 | 65.58 | 512 |
| hybrid + re-rank | DistilBERT (fine-tuned by us) | raw | 68.00 | 76.35 | 528 |
| hybrid + re-rank | DistilBERT (fine-tuned by us) | calibrated | 69.80 | 77.50 | 515 |
| hybrid + re-rank | Flan-T5 (flan-t5-base) | - | 75.80 | 82.91 | 931 |
| hybrid + re-rank | Ensemble (agreement / confidence) | calibrated | 75.20 | 82.37 | 1446 |

**C. Dual-reader agreement** (calibrated DistilBERT vs Flan-T5 on the same hybrid + re-ranked passages)

| Subset | Questions | DistilBERT EM | Flan-T5 EM |
|---|---:|---:|---:|
| readers agree (answer F1 >= 0.5) | 382 (76.4 %) | 83.8 | 84.6 |
| readers disagree | 118 | 24.6 | 47.5 |

Ensemble (agree -> extractive span, disagree -> the more confident reader): **EM 75.20 / F1 82.37**.

**D. Selective answering** - answer only the most confident questions:

| Coverage | Calibrated DistilBERT EM / F1 | Flan-T5 EM / F1 |
|---:|---:|---:|
| 100 % | 69.8 / 77.5 | 75.8 / 82.9 |
| 80 % | 78.8 / 84.7 | 85.5 / 89.8 |
| 60 % | 87.0 / 89.9 | 89.0 / 91.7 |
| 40 % | 94.0 / 94.6 | 93.5 / 94.6 |
