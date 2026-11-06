# Question Answering in LLMs

Retrieval-augmented question answering over SQuAD v1.1, built as the completion of the
summer 2025 research internship project *"Question Answering in LLMs"* (SRM University AP,
guide: Dr. M. Krishna Siva Prasad).

* **Retriever**: Sentence-Transformers (`all-MiniLM-L6-v2`) embeddings + FAISS exact cosine search over SQuAD passages
* **Reader 1 (extractive)**: `distilbert-base-cased-distilled-squad`, **fine-tuned by us** on SQuAD with the Hugging Face `Trainer`
* **Reader 2 (generative)**: `google/flan-t5-base`, zero-shot
* **Evaluation**: measured Exact Match, F1, Recall@k and MRR
* **Interfaces**: CLI (`python -m qa.cli`) and a FastAPI service with a small web page

```
question ─► MiniLM encoder ─► FAISS top-k ─► passages ─┬─► DistilBERT (span extraction) ─┐
                                                        └─► Flan-T5 (generation) ─────────┴─► answer + source
```

See [docs/architecture.md](docs/architecture.md) for the full diagram and design notes.

## Results

All numbers below were produced by `python -m qa.evaluate` on the machine noted, and are
written into this section automatically. Raw numbers: [results/results.json](results/results.json);
per-question predictions: `results/predictions_*.jsonl`.

<!-- RESULTS:START -->
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
<!-- RESULTS:END -->

**Fine-tuning run** ([results/training_log.json](results/training_log.json)) - CPU, same machine:

| Train examples (windows) | Epochs / steps | Batch | LR | Time | Train loss | Val. loss before -> after (300 val. questions) |
|---:|---:|---:|---:|---:|---:|---:|
| 3,000 (3,044) | 1 / 191 | 16 | 3e-5 | 37.4 min | 0.579 | 0.992 -> 0.989 |

### What the numbers say

1. **Our fine-tuning keeps the reader where it was rather than improving it.** The starting
   checkpoint was already distilled on SQuAD, so one more epoch on 3,000 SQuAD examples moves
   validation loss only from 0.992 to 0.989. With the gold passage, EM goes 79.1 -> 77.7 and
   F1 86.8 -> 86.3; in the full pipeline, F1 goes 65.1 -> 65.6. 109 of the 1,000 oracle
   predictions changed, mostly in span boundaries. Differences this small are within the noise of
   a 500-1,000-question sample. Clear gains would need a reader that has not already seen SQuAD
   (e.g. `distilbert-base-cased`), or a different domain.
2. **Retrieval is the main bottleneck in the full pipeline.** The gold paragraph is in the top 3
   for 80.5 % of all questions (77.2 % of the 500-question open sample). When it is retrieved,
   the readers score close to their oracle numbers (EM 72.3 base / 71.8 ours / 78.5 Flan-T5).
   When it is missed, EM falls to about 5-6. The remaining ~6-7 EM points come from distractor
   passages.
3. **Flan-T5-base is the stronger reader, but slower.** It leads by about 5 EM / 5 F1 with the gold
   passage and by about 5 EM / 3 F1 in the pipeline, at roughly 4x the CPU latency of DistilBERT
   (723 vs 190 ms per question with one passage). Part of its edge is likely because SQuAD is in
   Flan's instruction-tuning data.
4. **Extractive scores do not compare well across passages.** Raw start+end logits from different
   passages are not calibrated against each other. One example: for "Which NFL team won Super
   Bowl 50?", both DistilBERT readers answer *Carolina Panthers* (the losing team named in the
   same sentence) even with the gold passage alone, while Flan-T5 answers *Denver Broncos*.

**A bug the evaluation caught.** The first evaluation gave Flan-T5 only 22.8 EM with retrieved
passages. With three passages the prompt went past T5's 512-token limit, and right-truncation
cut off the question at the end of the prompt. The prompt builder now trims passages to fit, so
the question is always kept ([qa/readers.py](qa/readers.py)). All numbers above come from the
corrected run.

## Quick start

Requires Python 3.10+ (tested on 3.11, Windows 11, CPU only). No GPU needed.

```bash
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on Linux/macOS), then:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -r requirements.txt
```

(On a machine with an NVIDIA GPU, install the CUDA build of torch from pytorch.org instead of the CPU wheel.)

Run the whole pipeline in order:

```bash
python -m qa.train          # fine-tune DistilBERT -> models/distilbert-squad-finetuned
python -m qa.build_index    # embed 2,067 SQuAD passages -> index/
python -m qa.evaluate       # measure everything -> results/ and this README
pytest -q                   # unit + integration tests
```

Or in one go: `powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1` (Windows) /
`bash scripts/run_all.sh` (Linux/macOS).

Models and the dataset are downloaded from the Hugging Face Hub on first use
(~1.3 GB in total, cached in `~/.cache/huggingface`).

> **Windows long paths.** If the repo sits in a very deep folder and `pip install` fails
> with `ensurepip` / path errors, either enable Windows long-path support or create the
> venv at a short path (e.g. `py -3.11 -m venv C:\venvs\qa-llm`) and use that interpreter.

## Usage

### CLI

```bash
python -m qa.cli ask "Who was the Super Bowl 50 MVP?"
python -m qa.cli ask "Who was the Super Bowl 50 MVP?" --reader generative --top-k 5 --show-passages
python -m qa.cli compare "Which network broadcasted Super Bowl 50 in the United States?"     # both readers side by side
python -m qa.cli chat                                          # interactive loop
python -m qa.cli read -q "What is the capital of France?" -c "France, officially the French Republic, has its capital in Paris."
python -m qa.cli ask "Who designed the Apollo Lunar Module?" --json
```

### API

```bash
uvicorn qa.api:app --port 8000
```

* Web page: <http://127.0.0.1:8000>
* Swagger UI: <http://127.0.0.1:8000/docs>

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"Who was the Super Bowl 50 MVP?\", \"reader\": \"extractive\", \"top_k\": 3}"
```

| Endpoint | Method | Body / response |
|---|---|---|
| `/ask` | POST | `{"question": str, "reader": "extractive" \| "generative", "top_k": 1-10}` -> answer, score, source title, retrieved passages, latency |
| `/health` | GET | status, number of indexed passages, model names |
| `/` | GET | HTML demo page |

## Training (fine-tuning DistilBERT)

`python -m qa.train` fine-tunes `distilbert-base-cased-distilled-squad` with the HF `Trainer`
(AdamW, lr 3e-5, batch 16, 10 % warm-up, weight decay 0.01, 384-token windows with stride 128,
windows without the answer labelled with the `[CLS]` position). It logs validation loss on
held-out SQuAD validation questions before and after training and writes
`results/training_log.json`.

| Flag | CPU default | GPU default | Notes |
|---|---|---|---|
| `--train-samples` | 3,000 | 10,000 | random SQuAD train examples (seed 42) |
| `--epochs` | 1 | 3 | 3 epochs x 10k = report setting |
| `--eval-samples` | 300 | 1,000 | validation-loss subset |
| `--max-steps` | - | - | fixed step budget instead of epochs |

**CPU vs. Colab.** The CPU default trains in about 40 minutes on a laptop (37 min measured on an i5-1235U). The internship
report's full setting (10,000 examples x 3 epochs, fp16) is ~10x more compute and should be run
on a GPU: open [notebooks/train_on_colab.ipynb](notebooks/train_on_colab.ipynb) in Google Colab
with a T4 runtime, then unzip the downloaded model into `models/`. Everything else
(index, evaluation, CLI, API) runs comfortably on CPU.

## Evaluation protocol

* **Data**: SQuAD v1.1 (`rajpurkar/squad`). Fine-tuning uses only the *train* split; all
  evaluation uses the *validation* split (10,570 questions, 2,067 unique paragraphs).
* **Retrieval**: every validation question is searched against the 2,067 paragraphs; a hit
  means the question's own paragraph is in the top-k. Reported as Recall@1/3/5/10/20 and MRR@20.
* **Readers**, each in two settings:
  * *oracle passage* - the reader is given the gold paragraph (reading-comprehension ability);
  * *retrieved top-k* - the reader is given the top-3 retrieved paragraphs (the full RAG system).
* **Metrics**: SQuAD v1.1 Exact Match and token-level F1 with the official answer
  normalisation (lower-case, strip punctuation and articles), maximised over the reference
  answers ([qa/metrics.py](qa/metrics.py)).
* **Sample sizes** (CPU): retrieval on all 10,570 questions; readers on a fixed random sample
  of 1,000 (oracle) and 500 (retrieved) questions, seed 42. `--reader-samples 0` uses all questions.

Caveats worth knowing when reading the numbers:

* `distilbert-base-cased-distilled-squad` was already distilled on SQuAD, so further
  fine-tuning on SQuAD starts from a strong model; the comparison "pretrained vs. fine-tuned
  by us" is therefore reported side by side rather than assumed.
* Flan-T5's instruction-tuning mixture includes SQuAD, so its zero-shot score is not a clean
  out-of-domain number.
* The retrieval corpus is the SQuAD validation paragraphs only (closed corpus), not all of Wikipedia.

## Project structure

```
qa-llm/
├── qa/
│   ├── config.py        # model names, paths, sizes (env-overridable: QA_*)
│   ├── data.py          # SQuAD loading, passage corpus
│   ├── retriever.py     # Sentence-Transformers + FAISS
│   ├── readers.py       # DistilBERT extractive + Flan-T5 generative readers
│   ├── pipeline.py      # retrieve -> read
│   ├── train.py         # fine-tuning with HF Trainer
│   ├── build_index.py   # build FAISS index
│   ├── evaluate.py      # EM / F1 / Recall@k / MRR, writes results + README table
│   ├── metrics.py       # SQuAD metrics
│   ├── cli.py           # command-line interface
│   └── api.py           # FastAPI app + web page
├── tests/               # metric unit tests, pipeline/API integration tests
├── notebooks/train_on_colab.ipynb
├── scripts/run_all.ps1, run_all.sh
├── docs/architecture.md
├── results/             # results.json, results.md, training_log.json
├── requirements.txt
└── README.md
```

`models/` (fine-tuned weights, ~250 MB) and `index/` are generated and git-ignored.

## Configuration

Override any setting with an environment variable, e.g. a larger generator:

```bash
QA_GEN_MODEL=google/flan-t5-large python -m qa.evaluate --readers flan
```

Available: `QA_BASE_READER`, `QA_FINETUNED_READER_DIR`, `QA_GEN_MODEL`, `QA_EMBED_MODEL`,
`QA_INDEX_DIR`, `QA_RESULTS_DIR`, `QA_MAX_LENGTH`, `QA_DOC_STRIDE`, `QA_SEED`.

## Authors

G Sai Karthikeya (AP23110010631), Shaik Sameed (AP23110010598) - Department of CSE, SRM University AP.
