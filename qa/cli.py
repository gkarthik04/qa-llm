"""Command-line interface.

    python -m qa.cli ask "Who was the Super Bowl 50 MVP?"
    python -m qa.cli ask "Who was the Super Bowl 50 MVP?" --reader generative --top-k 5
    python -m qa.cli compare "Which network broadcasted Super Bowl 50 in the United States?"
    python -m qa.cli chat                      # interactive loop
    python -m qa.cli read -q "What is the capital of France?" -c "France ... capital in Paris."
"""
import argparse
import json
import sys
import textwrap

from transformers.utils import logging as hf_logging

from .pipeline import QAPipeline
from .readers import load_reader

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()


def _print_result(res, show_passages: bool):
    print(f"\nQ: {res.question}")
    print(f"A [{res.reader}]: {res.answer or '(no answer)'}")
    score = f"{res.score:.2f}" if res.score is not None else "n/a"
    print(f"   source: {res.source_title} | score: {score} | {res.latency_ms:.0f} ms")
    if show_passages:
        for i, hit in enumerate(res.retrieved, 1):
            print(f"\n  [{i}] {hit['title']} (sim {hit['score']:.3f})")
            print(textwrap.indent(textwrap.fill(hit["text"], 100), "      "))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qa", description="Retrieval-augmented QA over SQuAD passages")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ask = sub.add_parser("ask", help="answer one question with retrieval")
    p_ask.add_argument("question")
    p_ask.add_argument("--reader", choices=["extractive", "generative"], default="extractive")
    p_ask.add_argument("--top-k", type=int, default=3)
    p_ask.add_argument("--show-passages", action="store_true")
    p_ask.add_argument("--json", action="store_true", help="print the full result as JSON")

    p_cmp = sub.add_parser("compare", help="answer with both readers side by side")
    p_cmp.add_argument("question")
    p_cmp.add_argument("--top-k", type=int, default=3)

    p_chat = sub.add_parser("chat", help="interactive question loop")
    p_chat.add_argument("--reader", choices=["extractive", "generative"], default="extractive")
    p_chat.add_argument("--top-k", type=int, default=3)

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
    if args.cmd == "ask":
        res = pipe.ask(args.question, reader=args.reader, top_k=args.top_k)
        if args.json:
            print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
        else:
            _print_result(res, args.show_passages)
    elif args.cmd == "compare":
        for kind in ("extractive", "generative"):
            _print_result(pipe.ask(args.question, reader=kind, top_k=args.top_k), show_passages=False)
    elif args.cmd == "chat":
        print(f"Reader: {args.reader}. Type a question (empty line or 'exit' to quit).")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in {"exit", "quit"}:
                break
            _print_result(pipe.ask(q, reader=args.reader, top_k=args.top_k), show_passages=False)


if __name__ == "__main__":
    sys.exit(main())
