"""Measure retrieval (Recall@k, MRR) and reader quality (EM, F1) on SQuAD v1.1 validation.

Two reader settings are evaluated:
  * oracle  - the reader gets the gold paragraph (pure reading comprehension)
  * open    - the reader gets the top-k paragraphs from the FAISS retriever (full RAG pipeline)

    python -m qa.evaluate                       # CPU-friendly defaults
    python -m qa.evaluate --reader-samples 0    # all 10,570 questions (slow on CPU)
"""
import argparse
import json
import time
from pathlib import Path

import torch

from . import config
from .data import build_corpus, gold_passage_ids, load_squad, sample
from .metrics import mean_reciprocal_rank, recall_at_k, squad_scores
from .readers import ExtractiveReader, GenerativeReader
from .retriever import DenseRetriever

README_START = "<!-- RESULTS:START -->"
README_END = "<!-- RESULTS:END -->"


def evaluate_retrieval(retriever, ds, passages, ks):
    gold = gold_passage_ids(ds, passages)
    t0 = time.perf_counter()
    hits = retriever.search(ds["question"], k=max(ks))
    seconds = time.perf_counter() - t0
    ids = [[p.pid for p, _ in row] for row in hits]
    out = recall_at_k(ids, gold, ks)
    out["mrr@%d" % max(ks)] = 100.0 * mean_reciprocal_rank(ids, gold)
    out["n"] = len(ds)
    out["ms_per_query"] = 1000 * seconds / len(ds)
    return out


def run_reader(reader, questions, contexts, chunk=32):
    preds = []
    t0 = time.perf_counter()
    for i in range(0, len(questions), chunk):
        preds.extend(reader.read_batch(questions[i:i + chunk], contexts[i:i + chunk]))
        print(f"    {min(i + chunk, len(questions))}/{len(questions)}", end="\r", flush=True)
    print()
    return preds, 1000 * (time.perf_counter() - t0) / len(questions)


def write_predictions(path: Path, ds, preds):
    with open(path, "w", encoding="utf-8") as f:
        for q, refs, p in zip(ds["question"], ds["answers"], preds):
            f.write(json.dumps({"question": q, "gold": refs["text"], "prediction": p.text}, ensure_ascii=False) + "\n")


def results_markdown(res: dict) -> str:
    r = res["retrieval"]
    ks = res["settings"]["ks"]
    lines = [
        f"Measured on `{res['settings']['machine']}` (CPU only, {res['settings']['date']}).",
        "",
        f"**Retriever** - `{res['settings']['embed_model']}` + FAISS `IndexFlatIP` over "
        f"{res['settings']['num_passages']:,} SQuAD v1.1 validation paragraphs, "
        f"{r['n']:,} questions:",
        "",
        "| " + " | ".join(f"Recall@{k}" for k in ks) + f" | MRR@{max(ks)} | ms / query |",
        "|" + "---:|" * (len(ks) + 2),
        "| " + " | ".join(f"{r[f'recall@{k}']:.1f}" for k in ks) + f" | {r[f'mrr@{max(ks)}']:.1f} | {r['ms_per_query']:.1f} |",
        "",
        "**Readers** - SQuAD v1.1 validation, EM / F1 in %:",
        "",
        "| Reader | Type | Setting | Questions | EM | F1 | ms / question |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in res["readers"]:
        lines.append(f"| {row['reader']} | {row['type']} | {row['setting']} | {row['n']:,} | "
                     f"{row['exact_match']:.2f} | {row['f1']:.2f} | {row['ms_per_question']:.0f} |")
    return "\n".join(lines)


def update_readme(md: str, readme: Path = config.ROOT / "README.md"):
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    if README_START in text and README_END in text:
        head, rest = text.split(README_START, 1)
        _, tail = rest.split(README_END, 1)
        readme.write_text(f"{head}{README_START}\n{md}\n{README_END}{tail}", encoding="utf-8")
        print(f"Updated results section in {readme}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrieval-samples", type=int, default=0, help="0 = all validation questions")
    ap.add_argument("--reader-samples", type=int, default=1000, help="questions for the oracle setting (0 = all)")
    ap.add_argument("--open-samples", type=int, default=500, help="questions for the retrieve-then-read setting")
    ap.add_argument("--top-k", type=int, default=3, help="passages given to the reader in the open setting")
    ap.add_argument("--readers", default="base,finetuned,flan",
                    help="comma list from: base (pretrained DistilBERT), finetuned (ours), flan (Flan-T5)")
    ap.add_argument("--no-readme", action="store_true", help="do not write the results into README.md")
    args = ap.parse_args()
    ks = [1, 3, 5, 10, 20]

    torch.manual_seed(config.SEED)
    ds = load_squad("validation")
    try:
        retriever = DenseRetriever.load()
    except FileNotFoundError:
        print("Index not found - building it now")
        retriever = DenseRetriever()
        retriever.build(build_corpus(ds))
        retriever.save()
    passages = retriever.passages

    out_dir = config.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("== Retrieval")
    retrieval = evaluate_retrieval(retriever, sample(ds, args.retrieval_samples), passages, ks)
    print("   " + ", ".join(f"{k}={v:.2f}" for k, v in retrieval.items() if k != "n"))

    oracle_ds = sample(ds, args.reader_samples)
    open_ds = sample(ds, args.open_samples)
    open_hits = retriever.search(open_ds["question"], k=args.top_k)
    open_contexts = [[p.text for p, _ in row] for row in open_hits]

    readers = []
    wanted = [r.strip() for r in args.readers.split(",") if r.strip()]
    for key in wanted:
        if key == "base":
            readers.append(("base", "DistilBERT (pretrained, not fine-tuned by us)", "extractive",
                            lambda: ExtractiveReader(config.BASE_READER)))
        elif key == "finetuned":
            if not (config.FINETUNED_READER_DIR / "config.json").exists():
                print(f"!! No fine-tuned model at {config.FINETUNED_READER_DIR}; run `python -m qa.train`. Skipping.")
                continue
            readers.append(("finetuned", "DistilBERT (fine-tuned by us)", "extractive",
                            lambda: ExtractiveReader(str(config.FINETUNED_READER_DIR))))
        elif key == "flan":
            readers.append(("flan", f"Flan-T5 ({config.GEN_MODEL.split('/')[-1]}, zero-shot)", "generative",
                            lambda: GenerativeReader(config.GEN_MODEL)))
        else:
            raise SystemExit(f"Unknown reader key: {key}")

    rows = []
    for key, label, kind, factory in readers:
        reader = factory()
        for setting, sub, contexts in [
            ("oracle passage", oracle_ds, [[c] for c in oracle_ds["context"]]),
            (f"retrieved top-{args.top_k}", open_ds, open_contexts),
        ]:
            print(f"== {label} | {setting} | {len(sub)} questions")
            preds, ms = run_reader(reader, sub["question"], contexts)
            scores = squad_scores([p.text for p in preds], [a["text"] for a in sub["answers"]])
            print(f"   EM={scores['exact_match']:.2f}  F1={scores['f1']:.2f}  ({ms:.0f} ms/question)")
            rows.append({"reader": label, "key": key, "type": kind, "setting": setting,
                         "model": reader.model_path, "ms_per_question": ms, **scores})
            tag = "oracle" if setting.startswith("oracle") else "open"
            write_predictions(out_dir / f"predictions_{key}_{tag}.jsonl", sub, preds)
        del reader

    res = {
        "settings": {
            "date": time.strftime("%Y-%m-%d"),
            "machine": f"{config.cpu_name()}, {torch.get_num_threads()} threads",
            "torch": torch.__version__,
            "embed_model": retriever.model_name,
            "num_passages": len(passages),
            "ks": ks,
            "top_k": args.top_k,
            "reader_samples": len(oracle_ds),
            "open_samples": len(open_ds),
            "seed": config.SEED,
        },
        "retrieval": retrieval,
        "readers": rows,
    }
    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    md = results_markdown(res)
    (out_dir / "results.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)
    if not args.no_readme:
        update_readme(md)


if __name__ == "__main__":
    main()
