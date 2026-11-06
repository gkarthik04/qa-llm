"""Command-line interface.

    python -m qa.cli ask "Who was the Super Bowl 50 MVP?"
    python -m qa.cli ask "Who was the Super Bowl 50 MVP?" --reader generative --top-k 5 --show-passages
    python -m qa.cli ask "Who was the Super Bowl 50 MVP?" --retrieval dense --no-rerank --scoring raw
    python -m qa.cli compare "Which NFL team won Super Bowl 50?"        # both readers + agreement verdict
    python -m qa.cli ask "Who supervised the project?" --doc report.pdf  # question over your own file(s)
    python -m qa.cli chat                                                # interactive loop
    python -m qa.cli read -q "What is the capital of France?" -c "France ... capital in Paris."
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

from transformers.utils import logging as hf_logging

from . import config
from .pipeline import QAPipeline
from .readers import load_reader

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()


def _conf(c):
    return f"{c * 100:.0f}%" if c is not None else "n/a"


def _print_result(res, show_passages: bool):
    print(f"\nQ: {res.question}")
    print(f"A [{res.reader}]: {res.answer or '(no answer)'}")
    t = res.timings
    print(f"   source: {res.source_title} | confidence: {_conf(res.confidence)} | "
          f"retrieve {t.get('retrieve_ms', 0):.0f} ms, re-rank {t.get('rerank_ms', 0):.0f} ms, read {t.get('read_ms', 0):.0f} ms")
    alts = [a for a in res.alternatives[1:4] if a["confidence"] >= 0.02]
    if alts:
        print("   also considered: " + ", ".join(f"{a['text']} ({_conf(a['confidence'])})" for a in alts))
    if show_passages:
        for i, hit in enumerate(res.retrieved, 1):
            print(f"\n  [{i}] {hit['title']}  (P(rel) {hit['prior']:.2f}, cos {hit['dense'] or 0:.3f}, bm25 {hit['bm25'] or 0:.1f})")
            print(textwrap.indent(textwrap.fill(hit["text"], 100), "      "))


def _print_compare(out):
    e, g, a = out["extractive"], out["generative"], out["agreement"]
    print(f"\nQ: {out['question']}")
    print(f"A [extractive]: {e['answer'] or '(no answer)'}   (confidence {_conf(e['confidence'])})")
    print(f"A [generative]: {g['answer'] or '(no answer)'}   (confidence {_conf(g['confidence'])})")
    verdict = "AGREE - high confidence" if a["agree"] else "DISAGREE - check the evidence"
    print(f"   readers {verdict} (answer overlap F1 {a['f1']:.2f}) | {out['latency_ms']:.0f} ms")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qa", description="Retrieval-augmented QA over SQuAD passages or your own files")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def retrieval_args(p):
        p.add_argument("--top-k", type=int, default=3)
        p.add_argument("--retrieval", choices=["dense", "bm25", "hybrid"], default=config.DEFAULT_RETRIEVAL)
        p.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=config.DEFAULT_RERANK,
                       help="cross-encoder re-ranking of the top 20 (default: %(default)s)")
        p.add_argument("--scoring", choices=["raw", "calibrated"], default=config.DEFAULT_SCORING,
                       help="extractive span scoring (default: %(default)s)")
        p.add_argument("--doc", action="append", default=[], metavar="FILE",
                       help="answer from this PDF/TXT/MD file instead of SQuAD (repeatable)")

    p_ask = sub.add_parser("ask", help="answer one question with retrieval")
    p_ask.add_argument("question")
    p_ask.add_argument("--reader", choices=["extractive", "generative"], default="extractive")
    p_ask.add_argument("--show-passages", action="store_true")
    p_ask.add_argument("--json", action="store_true", help="print the full result as JSON")
    retrieval_args(p_ask)

    p_cmp = sub.add_parser("compare", help="answer with both readers and report whether they agree")
    p_cmp.add_argument("question")
    retrieval_args(p_cmp)

    p_chat = sub.add_parser("chat", help="interactive question loop")
    p_chat.add_argument("--reader", choices=["extractive", "generative", "compare"], default="extractive")
    retrieval_args(p_chat)

    p_read = sub.add_parser("read", help="answer from a context you supply (no retrieval)")
    p_read.add_argument("-q", "--question", required=True)
    p_read.add_argument("-c", "--context", required=True)
    p_read.add_argument("--reader", choices=["extractive", "generative"], default="extractive")

    args = ap.parse_args(argv)

    if args.cmd == "read":
        ans = load_reader(args.reader).read(args.question, [args.context])
        print(ans.text)
        return

    pipe = QAPipeline()
    corpus = "squad"
    for path in args.doc:
        doc = pipe.documents.add(Path(path).name, Path(path).read_bytes())
        print(f"Indexed {doc.name}: {doc.chunks} passages")
        corpus = "docs"
    kw = dict(top_k=args.top_k, retrieval=args.retrieval, rerank=args.rerank, scoring=args.scoring, corpus=corpus)

    if args.cmd == "ask":
        res = pipe.ask(args.question, reader=args.reader, **kw)
        if args.json:
            print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
        else:
            _print_result(res, args.show_passages)
    elif args.cmd == "compare":
        _print_compare(pipe.compare(args.question, **kw))
    elif args.cmd == "chat":
        print(f"Reader: {args.reader} | {args.retrieval}{' + re-rank' if args.rerank else ''} | corpus: {corpus}. "
              "Type a question (empty line or 'exit' to quit).")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in {"exit", "quit"}:
                break
            if args.reader == "compare":
                _print_compare(pipe.compare(q, **kw))
            else:
                _print_result(pipe.ask(q, reader=args.reader, **kw), show_passages=False)


if __name__ == "__main__":
    sys.exit(main())
