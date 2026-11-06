"""Measure the advanced features on SQuAD v1.1 validation (same seeded samples as qa.evaluate).

  A. Retrieval: dense vs BM25 vs hybrid (RRF) vs hybrid + cross-encoder re-ranking
  B. Readers on the full pipeline: raw vs calibrated extractive scoring, Flan-T5, and
     dual-reader agreement (ensemble + selective answering)

    python -m qa.evaluate_advanced
"""
import argparse
import json
import time

import numpy as np
import torch

from . import config
from .data import gold_passage_ids, load_squad, sample
from .metrics import exact_match, f1_score, max_over_truths, mean_reciprocal_rank, recall_at_k, squad_scores
from .readers import ExtractiveReader, GenerativeReader
from .retriever import DenseRetriever
from .search import SearchEngine, _softmax

START, END = "<!-- ADVANCED:START -->", "<!-- ADVANCED:END -->"


def run_retrieval(engine, questions, gold, mode, rerank, k):
    t0 = time.perf_counter()
    hits = []
    for i in range(0, len(questions), 50):  # chunks keep cross-encoder memory flat
        hits.extend(engine.search_batch(questions[i:i + 50], k=k, mode=mode, rerank=rerank, candidates=20))
        print(f"    {min(i + 50, len(questions))}/{len(questions)}", end="\r", flush=True)
    print()
    ms = 1000 * (time.perf_counter() - t0) / len(questions)
    ids = [[h.passage.pid for h in row] for row in hits]
    out = recall_at_k(ids, gold, [1, 3, 5, 10])
    out["mrr@10"] = 100 * mean_reciprocal_rank(ids, gold)
    out["ms_per_query"] = ms
    return out, hits


def top3(hits, rerank):
    """First 3 hits with the prior re-normalised over those 3 (as the pipeline does with k=3)."""
    rows = []
    for row in hits:
        row = row[:3]
        vals = np.array([h.rerank if rerank else h.dense for h in row])
        prior = _softmax(vals if rerank else vals / 0.05)
        rows.append(([h.passage.text for h in row], prior.tolist()))
    return rows


def read(reader, questions, rows, scoring=None, chunk=16):
    preds = []
    t0 = time.perf_counter()
    for i in range(0, len(questions), chunk):
        ctx = [r[0] for r in rows[i:i + chunk]]
        pri = [r[1] for r in rows[i:i + chunk]]
        preds.extend(reader.read_batch(questions[i:i + chunk], ctx, priors=pri, scoring=scoring))
        print(f"    {min(i + chunk, len(questions))}/{len(questions)}", end="\r", flush=True)
    print()
    return preds, 1000 * (time.perf_counter() - t0) / len(questions)


def per_question(preds, refs):
    em = np.array([max_over_truths(exact_match, p.text, r) for p, r in zip(preds, refs)])
    f1 = np.array([max_over_truths(f1_score, p.text, r) for p, r in zip(preds, refs)])
    return em, f1


def selective(conf, em, f1, coverages=(1.0, 0.8, 0.6, 0.4)):
    """Answer only the most-confident fraction of questions; report EM/F1 on those."""
    order = np.argsort(-np.asarray(conf))
    out = []
    for c in coverages:
        n = max(1, int(round(c * len(order))))
        sel = order[:n]
        out.append({"coverage": c, "n": n, "exact_match": 100 * em[sel].mean(), "f1": 100 * f1[sel].mean()})
    return out


def markdown(res):
    s = res["settings"]
    L = [f"Measured on `{s['machine']}` (CPU only, {s['date']}). Retrieval: {s['retrieval_samples']:,} questions; "
         f"readers: {s['reader_samples']:,} questions (the same seeded sample as the baseline table), top-3 passages.",
         "", "**A. Retrieval variants**", "",
         "| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | ms / query |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in res["retrieval"].items():
        L.append(f"| {name} | {r['recall@1']:.1f} | {r['recall@3']:.1f} | {r['recall@5']:.1f} | {r['recall@10']:.1f} "
                 f"| {r['mrr@10']:.1f} | {r['ms_per_query']:.0f} |")
    L += ["", "**B. Full pipeline (retrieve top-3 -> read)**", "",
          "| Retrieval | Reader | Scoring | EM | F1 | ms / question (read) |", "|---|---|---|---:|---:|---:|"]
    for r in res["readers"]:
        L.append(f"| {r['retrieval']} | {r['reader']} | {r['scoring']} | {r['exact_match']:.2f} | {r['f1']:.2f} "
                 f"| {r['ms_per_question']:.0f} |")
    a = res["agreement"]
    L += ["", "**C. Dual-reader agreement** (calibrated DistilBERT vs Flan-T5 on the same hybrid + re-ranked passages)", "",
          "| Subset | Questions | DistilBERT EM | Flan-T5 EM |", "|---|---:|---:|---:|",
          f"| readers agree (answer F1 >= 0.5) | {a['agree_n']} ({a['agree_rate']:.1f} %) | {a['agree_em_ext']:.1f} | {a['agree_em_gen']:.1f} |",
          f"| readers disagree | {a['disagree_n']} | {a['disagree_em_ext']:.1f} | {a['disagree_em_gen']:.1f} |",
          "", f"Ensemble (agree -> extractive span, disagree -> the more confident reader): "
              f"**EM {a['ensemble']['exact_match']:.2f} / F1 {a['ensemble']['f1']:.2f}**.",
          "", "**D. Selective answering** - answer only the most confident questions:", "",
          "| Coverage | Calibrated DistilBERT EM / F1 | Flan-T5 EM / F1 |", "|---:|---:|---:|"]
    for e, g in zip(res["selective"]["extractive_calibrated"], res["selective"]["generative"]):
        L.append(f"| {int(e['coverage'] * 100)} % | {e['exact_match']:.1f} / {e['f1']:.1f} | {g['exact_match']:.1f} / {g['f1']:.1f} |")
    return "\n".join(L)


def update_readme(md):
    readme = config.ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if START in text and END in text:
        head, rest = text.split(START, 1)
        _, tail = rest.split(END, 1)
        readme.write_text(f"{head}{START}\n{md}\n{END}{tail}", encoding="utf-8")
        print("Updated README advanced-results section")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrieval-samples", type=int, default=1000)
    ap.add_argument("--reader-samples", type=int, default=500)
    ap.add_argument("--no-readme", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(config.SEED)

    ds = load_squad("validation")
    retriever = DenseRetriever.load()
    engine = SearchEngine.from_retriever(retriever)
    rs = sample(ds, args.retrieval_samples)
    gold = gold_passage_ids(rs, retriever.passages)
    questions = rs["question"]

    retrieval, hits_by = {}, {}
    for name, mode, rr in [("Dense (MiniLM + FAISS)", "dense", False), ("BM25", "bm25", False),
                           ("Hybrid (RRF dense + BM25)", "hybrid", False),
                           ("Hybrid + cross-encoder re-rank (top 20)", "hybrid", True)]:
        print(f"== Retrieval: {name}")
        retrieval[name], hits_by[(mode, rr)] = run_retrieval(engine, questions, gold, mode, rr, k=10)
        print("   " + ", ".join(f"{k}={v:.1f}" for k, v in retrieval[name].items()))

    n = min(args.reader_samples, len(rs))
    q = questions[:n]
    refs = [a["text"] for a in rs["answers"][:n]]
    dense_rows = top3(hits_by[("dense", False)][:n], rerank=False)
    best_rows = top3(hits_by[("hybrid", True)][:n], rerank=True)

    ext = ExtractiveReader(str(config.FINETUNED_READER_DIR) if (config.FINETUNED_READER_DIR / "config.json").exists()
                           else config.BASE_READER)
    readers, preds = [], {}
    for label, rows, scoring in [("dense", dense_rows, "raw"), ("hybrid + re-rank", best_rows, "raw"),
                                 ("hybrid + re-rank", best_rows, "calibrated")]:
        print(f"== DistilBERT ({scoring}) on {label}")
        p, ms = read(ext, q, rows, scoring)
        preds[(label, scoring)] = p
        readers.append({"retrieval": label, "reader": "DistilBERT (fine-tuned by us)", "scoring": scoring,
                        "ms_per_question": ms, **squad_scores([x.text for x in p], refs)})
        print(f"   EM={readers[-1]['exact_match']:.2f} F1={readers[-1]['f1']:.2f}")
    del ext

    gen = GenerativeReader(config.GEN_MODEL)
    print("== Flan-T5 on hybrid + re-rank")
    gp, ms = read(gen, q, best_rows)
    readers.append({"retrieval": "hybrid + re-rank", "reader": f"Flan-T5 ({config.GEN_MODEL.split('/')[-1]})",
                    "scoring": "-", "ms_per_question": ms, **squad_scores([x.text for x in gp], refs)})
    print(f"   EM={readers[-1]['exact_match']:.2f} F1={readers[-1]['f1']:.2f}")

    ep = preds[("hybrid + re-rank", "calibrated")]
    e_em, e_f1 = per_question(ep, refs)
    g_em, g_f1 = per_question(gp, refs)
    agree = np.array([(f1_score(a.text, b.text) >= 0.5) if a.text and b.text else False for a, b in zip(ep, gp)])
    e_conf = np.array([a.confidence or 0.0 for a in ep])
    g_conf = np.array([a.confidence or 0.0 for a in gp])
    pick_ext = agree | (e_conf >= g_conf)
    ens_em = np.where(pick_ext, e_em, g_em)
    ens_f1 = np.where(pick_ext, e_f1, g_f1)
    agreement = {
        "agree_n": int(agree.sum()), "agree_rate": 100 * agree.mean(), "disagree_n": int((~agree).sum()),
        "agree_em_ext": 100 * e_em[agree].mean(), "agree_em_gen": 100 * g_em[agree].mean(),
        "disagree_em_ext": 100 * e_em[~agree].mean(), "disagree_em_gen": 100 * g_em[~agree].mean(),
        "ensemble": {"exact_match": 100 * ens_em.mean(), "f1": 100 * ens_f1.mean()},
    }
    readers.append({"retrieval": "hybrid + re-rank", "reader": "Ensemble (agreement / confidence)", "scoring": "calibrated",
                    "ms_per_question": readers[-2]["ms_per_question"] + readers[-1]["ms_per_question"],
                    "exact_match": agreement["ensemble"]["exact_match"], "f1": agreement["ensemble"]["f1"], "n": n})

    with open(config.RESULTS_DIR / "predictions_advanced.jsonl", "w", encoding="utf-8") as f:
        for qq, r, a, b, ag in zip(q, refs, ep, gp, agree):
            f.write(json.dumps({"question": qq, "gold": r, "extractive": a.text, "extractive_conf": a.confidence,
                                "generative": b.text, "generative_conf": b.confidence, "agree": bool(ag)},
                               ensure_ascii=False) + "\n")

    res = {
        "settings": {"date": time.strftime("%Y-%m-%d"), "machine": f"{config.cpu_name()}, {torch.get_num_threads()} threads",
                     "retrieval_samples": len(rs), "reader_samples": n, "rerank_model": config.RERANK_MODEL,
                     "rerank_candidates": 20, "seed": config.SEED},
        "retrieval": retrieval,
        "readers": readers,
        "agreement": agreement,
        "selective": {"extractive_calibrated": selective(e_conf, e_em, e_f1), "generative": selective(g_conf, g_em, g_f1)},
    }
    (config.RESULTS_DIR / "advanced.json").write_text(json.dumps(res, indent=2))
    md = markdown(res)
    (config.RESULTS_DIR / "advanced.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)
    if not args.no_readme:
        update_readme(md)


if __name__ == "__main__":
    main()
