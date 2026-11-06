"""Embed the SQuAD passage corpus and write a FAISS index to ./index.

    python -m qa.build_index
"""
import argparse
import time

from . import config
from .data import build_corpus, load_squad
from .retriever import DenseRetriever


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=config.CORPUS_SPLIT)
    ap.add_argument("--embed-model", default=config.EMBED_MODEL)
    ap.add_argument("--out", default=str(config.INDEX_DIR))
    args = ap.parse_args()

    passages = build_corpus(load_squad(args.split))
    print(f"Corpus: {len(passages)} unique passages from SQuAD/{args.split}")
    t0 = time.time()
    retriever = DenseRetriever(args.embed_model)
    retriever.build(passages)
    retriever.save(args.out)
    print(f"Indexed {len(passages)} passages with {args.embed_model} in {time.time() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
